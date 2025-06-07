# Import necessary libraries
import pandas as pd                # For data manipulation and DataFrame operations
import numpy as np                 # Commonly used for numerical operations (not used directly here)
from faker import Faker            # To generate realistic fake data for testing
import random                     # To make random choices and generate random numbers
from datetime import datetime    # To work with dates and timestamps
import pyarrow as pa             # Apache Arrow library, used for in-memory columnar data representation
import pyarrow.parquet as pq     # For reading/writing Parquet files, an efficient columnar storage format
import io                        # Provides in-memory byte streams for file operations
import json                      # To serialize and deserialize JSON data
from google.cloud import storage # Google Cloud Storage client library to interact with GCS buckets
import os                        # To interact with environment variables and file paths
from dotenv import load_dotenv   # Loads environment variables from a .env file for security


# Load environment variables from a .env file to keep sensitive data (like API keys) out of code
load_dotenv()

# Retrieve the path to the Google Cloud service account JSON key from environment variables
keyfile_path = os.getenv('GCP_KEYFILE_PATH')

# Initialize Google Cloud Storage client with service account credentials
storage_client = storage.Client.from_service_account_json(keyfile_path)


# List and print all available buckets in the connected GCS project for verification
buckets = list(storage_client.list_buckets())
print("Successfully connected to GCS. Available buckets:")
for bucket in buckets:
    print(f" - {bucket.name}")

# Initialize Faker object to generate fake data
fake = Faker()

# Define constants for the bucket and folder paths for dev and prod environments
BUCKET_NAME = 'healthcare-data-bucket-treadway'
DEV_PATH = 'dev/'
PROD_PATH = 'prod/'

# Define number of records to generate for dev and prod datasets
DEV_RECORDS = 5000
PROD_RECORDS = 20000

# Define date range for generating realistic date values
start_date = datetime(2020, 1, 1)
end_date = datetime.today()


def create_bucket():
    """
    Create the specified GCS bucket if it does not already exist.
    Handles exceptions gracefully.
    """
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        if not bucket.exists():
            storage_client.create_bucket(BUCKET_NAME)
            print(f"Bucket '{BUCKET_NAME}' created successfully.")
        else:
            print(f"Bucket '{BUCKET_NAME}' already exists.")
    except Exception as e:
        print(f"Error creating bucket: {e}")


def empty_gcs_folder(path):
    """
    Delete all objects/blobs within the given folder path inside the GCS bucket.
    This is useful to clear out old data before uploading fresh datasets.
    """
    print(f"Emptying folder '{path}' in GCS bucket '{BUCKET_NAME}'...")
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=path)
    for blob in blobs:
        blob.delete()
        print(f"Deleted '{blob.name}'")
    print(f"Completed emptying folder '{path}'.")


def upload_to_gcs(data, path, filename, file_format):
    """
    Upload given data to Google Cloud Storage at the specified path and filename.
    Supports CSV, JSON (newline delimited), and Parquet formats.
    """
    print(f"Uploading {filename} in {file_format} format to GCS...")
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(path + filename)

    if file_format == 'csv':
        # Convert DataFrame to CSV string and upload
        csv_data = data.to_csv(index=False)
        blob.upload_from_string(csv_data, content_type='text/csv')
    elif file_format == 'json':
        # Upload JSON data as newline-delimited string
        json_data = "\n".join(data)
        blob.upload_from_string(json_data, content_type='application/json')
    elif file_format == 'parquet':
        # Write Apache Arrow Table to in-memory bytes buffer and upload as Parquet
        buffer = io.BytesIO()
        pq.write_table(data, buffer)
        buffer.seek(0)
        blob.upload_from_file(buffer, content_type='application/octet-stream')
    print(f"Uploaded {filename} to {path}")


def generate_patients(num_records):
    """
    Generate a DataFrame of fake patient demographic data including:
    patient_id, name, age, gender, zip code, insurance, registration date.
    """
    print("Generating patient demographic data in CSV format...")
    patients = []
    for _ in range(num_records):
        patient_id = fake.unique.uuid4()  # Unique identifier
        first_name = fake.first_name()
        last_name = fake.last_name()
        age = random.randint(0, 100)
        gender = random.choice(['Male', 'Female'])
        zip_code = fake.zipcode()
        insurance_type = random.choice(['Private', 'Medicare', 'Medicaid'])
        registration_date = fake.date_between(start_date=start_date, end_date=end_date)

        patients.append({
            'patient_id': patient_id,
            'first_name': first_name,
            'last_name': last_name,
            'age': age,
            'gender': gender,
            'zip_code': zip_code,
            'insurance_type': insurance_type,
            'registration_date': str(registration_date)
        })
    return pd.DataFrame(patients)


def generate_ehr(num_records, patient_ids):
    """
    Generate Electronic Health Records (EHR) data in newline-delimited JSON format.
    Each record contains patient visit info including diagnosis and vital signs.
    """
    print("Generating electronic health records data in newline-delimited JSON format...")
    ehr_records = []
    for _ in range(num_records):
        patient_id = random.choice(patient_ids)
        visit_date = fake.date_between(start_date=start_date, end_date=end_date)
        diagnosis_code = random.choice(['E11.9', 'I10', 'J45', 'N18.9', 'Z00.0'])
        diagnosis_desc = {
            'E11.9': 'Type 2 diabetes mellitus',
            'I10': 'Essential hypertension',
            'J45': 'Asthma',
            'N18.9': 'Chronic kidney disease',
            'Z00.0': 'General medical exam'
        }[diagnosis_code]
        heart_rate = random.randint(60, 100)
        blood_pressure = f"{random.randint(110, 140)}/{random.randint(70, 90)}"
        temperature = round(random.uniform(97.0, 99.5), 1)

        ehr_record = {
            'patient_id': patient_id,
            'visit_date': str(visit_date),
            'diagnosis_code': diagnosis_code,
            'diagnosis_desc': diagnosis_desc,
            'heart_rate': heart_rate,
            'blood_pressure': blood_pressure,
            'temperature': temperature
        }

        # Append JSON string representation of record for newline-delimited JSON
        ehr_records.append(json.dumps(ehr_record))

    return ehr_records


def generate_claims(num_records, patient_ids):
    """
    Generate healthcare claims data in Parquet format with an explicit schema.
    Each record includes claim ID, patient, provider, service date, diagnosis,
    procedure code, claim amount, and status.
    """
    print("Generating claims data in Parquet format with explicit schema...")
    claims = []
    for _ in range(num_records):
        claim_id = fake.unique.uuid4()
        patient_id = random.choice(patient_ids)
        provider_id = fake.unique.uuid4()
        service_date = fake.date_between(start_date=start_date, end_date=end_date)
        # Ensure service_date is a datetime object with time set to midnight
        service_date = datetime.combine(service_date, datetime.min.time())
        diagnosis_code = random.choice(['E11.9', 'I10', 'J45', 'N18.9'])
        procedure_code = random.choice(['99213', '80053', '83036', '93000'])
        claim_amount = round(random.uniform(100, 5000), 2)
        status = random.choice(['Paid', 'Denied', 'Pending'])

        claims.append({
            'claim_id': str(claim_id),
            'patient_id': str(patient_id),
            'provider_id': str(provider_id),
            'service_date': service_date,
            'diagnosis_code': diagnosis_code,
            'procedure_code': procedure_code,
            'claim_amount': float(claim_amount),
            'status': status
        })

    # Define explicit schema to enforce types in the Parquet file
    schema = pa.schema([
        ('claim_id', pa.string()),
        ('patient_id', pa.string()),
        ('provider_id', pa.string()),
        ('service_date', pa.timestamp('ms')),
        ('diagnosis_code', pa.string()),
        ('procedure_code', pa.string()),
        ('claim_amount', pa.float64()),
        ('status', pa.string())
    ])

    # Convert list of dicts to DataFrame and then to Apache Arrow Table with schema
    table = pa.Table.from_pandas(pd.DataFrame(claims), schema=schema)
    return table


# -------------------- Main script execution --------------------

# Ensure the GCS bucket exists before uploading data
create_bucket()

# Clear out development folder in GCS before uploading fresh dev data
empty_gcs_folder(DEV_PATH)
# Generate fake patient, EHR, and claims data for development environment
dev_patients = generate_patients(DEV_RECORDS)
dev_ehr = generate_ehr(DEV_RECORDS, dev_patients['patient_id'].tolist())
dev_claims = generate_claims(DEV_RECORDS, dev_patients['patient_id'].tolist())

# Upload generated development data to respective folders in GCS
upload_to_gcs(dev_patients, DEV_PATH, 'patient_data.csv', 'csv')
upload_to_gcs(dev_ehr, DEV_PATH, 'ehr_data.json', 'json')
upload_to_gcs(dev_claims, DEV_PATH, 'claims_data.parquet', 'parquet')

# Clear out production folder in GCS before uploading fresh prod data
empty_gcs_folder(PROD_PATH)
# Generate fake patient, EHR, and claims data for production environment
prod_patients = generate_patients(PROD_RECORDS)
prod_ehr = generate_ehr(PROD_RECORDS, prod_patients['patient_id'].tolist())
prod_claims = generate_claims(PROD_RECORDS, prod_patients['patient_id'].tolist())

# Upload generated production data to respective folders in GCS
upload_to_gcs(prod_patients, PROD_PATH, 'patient_data.csv', 'csv')
upload_to_gcs(prod_ehr, PROD_PATH, 'ehr_data.json', 'json')
upload_to_gcs(prod_claims, PROD_PATH, 'claims_data.parquet', 'parquet')

print("Data generation and upload to GCS completed successfully.")
