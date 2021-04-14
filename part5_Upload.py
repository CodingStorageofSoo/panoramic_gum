from minio import Minio
import os 
import argparse

# Argument  

parser = argparse.ArgumentParser(description="Upload the result to the site")
parser.add_argument('--base', '--b', help = 'the base address of the data')
parser.add_argument('--address', '--m', help = 'the address of MinIO')
parser.add_argument('--id', '--i', help = 'ID')
parser.add_argument('--password', '--p', help = 'Password')
args = parser.parse_args()

Basic=args.base
AD=args.address
ID=args.id
PS=args.password

# Connect the MinIO server

client = Minio(AD, access_key=ID, secret_key=PS, secure=True)

# Upload the result

print("Upload the result to Minio")

file_path = os.path.join(Basic, "result.xlsx")
minIO_address = "pano_result/result.xlsx"
client.fput_object("soo", minIO_address, file_path)

print ()
