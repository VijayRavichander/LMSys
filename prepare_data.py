from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

api = KaggleApi()
api.authenticate()

competition_name = "lmsys-chatbot-arena"  # Replace with the actual competition name
api.competition_download_files(competition_name, path="./data")

def unzip_file(zip_file_path, extract_to):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Example usage:
zip_file_path = './data/lmsys-chatbot-arena.zip'
extract_to = './data'
unzip_file(zip_file_path, extract_to)