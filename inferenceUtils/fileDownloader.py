import os
import re
import shutil
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import boto3
import requests
from tqdm import tqdm

from inferenceUtils.logSetting import logger

chunk_size = 8192
ignore_list = ['__MACOSX', 'Thumbs.db', 'desktop.ini']


@dataclass
class FileDownloadConfig(object):
    cache_dir: str
    timeout: int = 600
    num_proc: int = 1
    merge_block: int = 65536

    def __post_init__(self):
        if self.num_proc >= 32:
            logger.warning(
                "due to the limit with network speed ,now  num_proc should be less than 32, set num_proc to 32")
            self.num_proc = 32
        if self.timeout <= 300:
            logger.warning(f"timeout is set to {self.timeout}s, make sure it's enough for download your dataset")
            self.timeout = 300
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)


class FileDownloader(object):

    def __init__(self, config: FileDownloadConfig):
        self.download_config = config

    def _get_chunk_size_by_url(self, presigned_url):
        response = requests.get(presigned_url, stream=True, timeout=self.download_config.timeout)
        response.raise_for_status()
        block_size, content_length, file_path = self._info_prepare(presigned_url, response)
        return block_size, content_length, file_path

    def _get_chunk_size_by_s3(self, model_bucket_name, object_key):
        s3 = boto3.client('s3')
        response = s3.head_object(Bucket=model_bucket_name, Key=object_key)
        content_length = int(response['ContentLength'])
        block_size, file_path = self.prepare_progress(content_length, "", object_key)
        return block_size, content_length, file_path

    def download_to_temp_file(self, presigned_url=None, bucket=None, key=None):
        futures = []
        if presigned_url:
            block_size, content_length, file_path = self._get_chunk_size_by_url(presigned_url)
        elif bucket and key:
            block_size, content_length, file_path = self._get_chunk_size_by_s3(bucket, key)
        else:
            raise Exception("no url or bucket and key passed")
        with ThreadPoolExecutor(max_workers=self.download_config.num_proc) as executor:
            for i in range(self.download_config.num_proc):
                start = block_size * i
                end = block_size * (i + 1) - 1 if i != self.download_config.num_proc - 1 else content_length
                if bucket and key:
                    futures.append(
                        executor.submit(self._download_chunk_by_sdk, file_path, i, bucket, key, start, end))
                elif presigned_url:
                    futures.append(executor.submit(self._download_chunk, file_path, i, presigned_url, start, end))
        for future in futures:
            future.result()
        self._combine_files(file_path, self.download_config.num_proc)
        return file_path

    def extract_archive(self, file_path):
        logger.info(f"[NetMind]   Unzipping {file_path}")
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(path=os.path.dirname(file_path))
                non_ignored_paths = [name for name in zip_ref.namelist() if
                                     not any(ignore in name for ignore in ignore_list)]
                root_folder = os.path.commonpath(non_ignored_paths)
        else:
            if file_path.endswith(".tar"):
                open_mode = "r"
            elif file_path.endswith(".tar.gz"):
                open_mode = "r:gz"
            elif file_path.endswith(".tar.bz2"):
                open_mode = "r:bz2"
            else:
                raise Exception(f"Unsupported file type: {file_path}")
            with tarfile.open(file_path, open_mode) as tar_ref:
                tar_ref.extractall(path=os.path.dirname(file_path))
                non_ignored_paths = [name for name in tar_ref.namelist() if
                                     not any(ignore in name for ignore in ignore_list)]
                root_folder = os.path.commonpath(non_ignored_paths)

        if root_folder:
            extracted_folder = os.path.join(os.path.dirname(file_path), root_folder)
            logger.info(f"extracted_folder: {extracted_folder}")
            self._exrtact_common_folder(extracted_folder, file_path)
        os.remove(file_path)
        with open(os.path.join(os.path.dirname(file_path), "flag.conf"), 'w') as file:
            file.write("finished")
        return os.path.dirname(file_path)

    def _exrtact_common_folder(self, extracted_folder, file_path):
        if os.path.exists(extracted_folder) and os.path.isdir(extracted_folder):
            logger.info("do extracted_folder")
            for file_name in os.listdir(extracted_folder):
                os.rename(os.path.join(extracted_folder, file_name),
                          os.path.join(os.path.dirname(file_path), file_name))
            os.rmdir(extracted_folder)

    def _info_prepare(self, presigned_url, response):
        filenames = re.findall("https://.*?\.amazonaws\.com/((?:[\w-]+/)*)([\w-]+\.\w+(?:\.\w+)?)", presigned_url)
        # give default name with content type if no filename can be extract in url
        content_length = int(response.headers.get('content-length', 0))
        if filenames and len(filenames[0]) > 1:
            filename = filenames[0][1]
            common_path = filenames[0][0]
        else:
            content_type = response.headers.get('Content-Type', '').split('/')[-1]
            if content_type not in ["zip", "tar", "gz", "bz2"]:
                content_type = "zip"
            filename = f"downloaded_file.{content_type}"
            common_path = "common_user_id"
        block_size, file_path = self.prepare_progress(content_length, common_path, filename)
        return block_size, content_length, file_path

    def prepare_progress(self, content_length, user_id, filename):
        file_name_without_ext, ext = os.path.splitext(filename)
        folder_path = os.path.join(self.download_config.cache_dir, user_id, file_name_without_ext)
        try:
            os.makedirs(folder_path)
        except FileExistsError:
            logger.warning("folder already exists but without success flag, will overwrite it")
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
        file_path = os.path.join(self.download_config.cache_dir, user_id, file_name_without_ext, filename)
        # last block may be bigger than block_size , set last block end number to content_length
        block_size = content_length // self.download_config.num_proc
        return block_size, file_path

    def _download_chunk(self, file_path, file_number, url, start, end):
        temp_file_path = f"{file_path}.{file_number}"
        headers = {'Range': f'bytes={start}-{end}'}
        response = requests.get(url, headers=headers, stream=True, timeout=self.download_config.timeout)
        response.raise_for_status()
        pbar = tqdm(total=int(end) - int(start) + 1, unit='B', unit_scale=True,
                    desc=f"[NetMind]  Dataset Downloading [Thread-{file_number + 1}] ", miniters=100)
        with open(temp_file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))

    def _download_chunk_by_sdk(self, file_path, file_number, bucket_name, object_key, start, end):
        s3 = boto3.client('s3')
        range_header = f'bytes={start}-{end}'
        output_filename = f"{file_path}.{file_number}"
        pbar = tqdm(total=int(end) - int(start) + 1, unit='B', unit_scale=True,
                    desc=f"[NetMind] Downloading [Thread-{file_number + 1}] ", miniters=100)
        response = s3.get_object(Bucket=bucket_name, Key=object_key, Range=range_header)

        with open(output_filename, 'wb') as f:
            for chunk in response['Body'].iter_chunks(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    def _combine_files(self, file_path, num_proc):
        with open(file_path, 'wb') as outfile:
            for i in range(num_proc):
                temp_file_path = f"{file_path}.{i}"
                with open(temp_file_path, 'rb') as infile:
                    while True:
                        # merge in stream in case of large files
                        chunk = infile.read(self.download_config.merge_block)
                        if not chunk:
                            break
                        outfile.write(chunk)
                os.remove(temp_file_path)
