import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile


def unzip_file(zip_file_path, extract_to):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

api = KaggleApi()
api.authenticate()

competition_name = "lmsys-chatbot-arena"  
api.competition_download_files(competition_name, path="./data")

kaggle.api.kernels_output(
    "abdullahmeda/33k-lmsys-chatbot-arena-conversations",
    path="/data/extra-lmsys-data.zip",
    quiet=True
)

# Example usage:
zip_file_path = './data/lmsys-chatbot-arena.zip'
extract_to = './data'
unzip_file(zip_file_path, extract_to)


# Extract Suppl Data
zip_file_path = './data/extra-lmsys-data.zip'
extract_to = './data'
unzip_file(zip_file_path, extract_to)
