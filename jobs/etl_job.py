# /opt/jobs/etl_job.py

# 1. 모듈가져오기
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# 2. SparkSession 획득 혹은 생성 (클러스터 연결)
spark = SparkSession.builder \
.appName('Hospital_ETL_Cluster') \
.master('spark://spark-master:7077') \
.config("spark.executor.memory","512m") \
.config("spark.executor.cores","1") \
.config("spark.sql.shuffle.partitions","8") \
.getOrCreate()
# 로그 레벨 지정
spark.sparkContext.setLogLevel("WARN")

print("\n" + "-"*50)
print("클러스터 연결 정보")
print("-"*50)
print(f" Master   : { spark.sparkContext.master }")
print(f" App ID   : { spark.sparkContext.applicationId }")
print(f" App Name : { spark.sparkContext.appName }")
print("-"*50 + "\n")

# 3. Extract
# csv => Dataframe 구성
# 3-1. 스키마 정의
schema = StructType([
    StructField("visit_id",   LongType(),   False),
    StructField("patient_id", StringType(), False),# csv에 맞춰 컬럼명 조정
    StructField("visit_date", StringType(), True),
    StructField("department", StringType(), True), # 결측존재 -> null 데이터가 존재함 구성
    StructField("diagnosis",  StringType(), True),
    StructField("charge",     LongType(),   True), # 음수 데이터가 존재
    StructField("status",     StringType(), True)  # 취소건 존재
])
# 3-2. 데이터프레임(RDD + 스키마 + 최적화) 구성
#      헤더 존재(1번라인 헤더(컬럼임)), 스키마 지정, 로컬위치지정(향후 s3등 조정)
raw_df = spark.read \
.option("header", True) \
.schema(schema) \
.csv('/opt/data/hospital_raw.csv')

print("\n" + "-"*50)
print("[Extract] 데이터프레임 구성")
print("-"*50)
print(f"[Extract] : 원시 데이터 수 : { raw_df.count() }")
print(f"[Extract] : 파티션 수 : { raw_df.rdd.getNumPartitions() }")
print("-"*50 + "\n")

# 실제 데이터중 상위 20개 출력 -> 20개만 로드가됨
raw_df.show() # 액션 => 데이터가 파티션에 배치, 로드, 처리

# 4. Trasform => 워커내에서 작동(실제 개별 노드들에서 수행)
# 4-1. 불량 데이터(노이즈, 결측치, 음수값등) 처리
clean_df = ( raw_df
.filter( F.col('status') == '완료')
.filter( F.col('department').isNotNull() )
.filter( F.col('charge') > 0 )
)
print(f"[Transform - 1] 데이터 정제 후 : { clean_df.show() }")

# 4-2. 파생 변수(컬럼) 생성, 타입변경, 연도, 월, 쿼터, 진료비기반구간화
tranformed_df = ( clean_df
  .withColumn('visit_date', F.to_date('visit_date','yyyy-MM-dd'))
  .withColumn('visit_year', F.year('visit_date') ) # 년정보 컬럼 추가
  .withColumn('visit_month', F.month('visit_date')) # 월정보 컬럼
  .withColumn('visit_quarter',
      F.when(F.col('visit_month').between(1,3), "Q1")    # 1분기
       .when(F.col('visit_month').between(4,6), "Q2")    # 2분기
       .when(F.col('visit_month').between(7,9), "Q3")    # 3분기
       .otherwise("Q4") ) # 분기정보 컬럼
  .withColumn('charge_tier',
      F.when(F.col('charge') < 50000, "외래")
       .when(F.col('charge') < 200000, "일반")
       .otherwise("고액")
      )
)
print(f"[Transform - 2] 파생 컬럼 생성 : { tranformed_df.show() }")

# 4-3. Window 함수를 이용한 순서처리, 이전진료과, 누적진료비 추가
win_partition = Window.partitionBy('patient_id').orderBy('visit_date')
tans_win_df = ( tranformed_df
.withColumn("visit_seq", F.row_number().over(win_partition))
.withColumn("prev_department", F.lag("department", 1).over(win_partition))
.withColumn("cum_charge", F.sum("charge").over(win_partition))
)
print(f"[Transform - 3] 윈도우 함수 적용 후 : { tans_win_df.show() }")

# 보고 싶은 컬럼만 추출 (생략 가능)
tans_win_df.select("patient_id","visit_date","department",
                   "charge","visit_seq","cum_charge").show()

# 4-4. 파티션 분산 확인
print("파티션별로 데이터가 몇개 배치되었는지 모니터링")
tans_win_df.withColumn("partition_id", F.spark_partition_id()) \
.groupBy("partition_id") \
.count() \
.orderBy("partition_id") \
.show()

# 5. Load
department_monthly = (tans_win_df
  .groupBy('department', 'visit_year', 'visit_month') # 결과셋에 포한되는 컬럼(집계대상컬럼, 집계함수결과물)
  .agg(
    F.count('visit_id').alias('visit_count'), # 집계함수(컬럼명).alias(파생변수명)
    F.sum('charge').alias('total_charge'),
    F.avg('charge').alias('avg_charge'),
    F.countDistinct("patient_id").alias('unique_person')
  )
  .orderBy('department', 'visit_year', 'visit_month')
)
person_summary_df = (tans_win_df
  .groupBy('patient_id')
  .agg(
    F.count('visit_id').alias('total_visits'),
    F.sum('charge').alias('total_charge'),
    F.min('visit_date').alias('first_visit'),
    F.max('visit_date').alias('last_visit'),
    F.collect_set('department').alias('visited_departments') # 중복 제거
  )
  .orderBy('patient_id')
)

print("\n" + "-"*50)
print(f"[Load] (집계) 부서별 월간 실적 테이블 저장 {department_monthly.show()}")
print("-"*50)
print(f"[Load] (집계) 환자별 요약 테이블 {person_summary_df.show()}")
print("-"*50 + "\n")

print("파티션별로 데이터가 몇개 배치되었는지 모니터링")
department_monthly.withColumn("partition_id", F.spark_partition_id()) \
.groupBy("partition_id") \
.count() \
.orderBy("partition_id") \
.show()

person_summary_df.withColumn("partition_id", F.spark_partition_id()) \
.groupBy("partition_id") \
.count() \
.orderBy("partition_id") \
.show()

# 저장 -> 실제는 s3에 저장
# master에서 저장코드가 수행됨
department_monthly.write \
.partitionBy('visit_year', 'visit_month') \
.mode('overwrite') \
.parquet('/tmp/gold/output/dept_monthly')

person_summary_df.write \
.mode('overwrite') \
.parquet('/tmp/gold/output/person_summary')

print("\n" + "-"*50)
print("ETL 완료")
print("-"*50 + "\n")

# 6. 스파크 종료(세션 종료)
spark.stop()

print("\n" + "-"*50)
print("스파크 종료(세션 종료)")
print("-"*50 + "\n")