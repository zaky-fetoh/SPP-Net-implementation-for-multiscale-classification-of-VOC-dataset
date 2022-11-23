import tarfile as tar
import urllib.request
from tqdm import tqdm
import os

dataset_url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
file_name ='./data_org/voc.tar'
xoutpath = './data_org/'

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def dataset_downloader(url=dataset_url, output_path=file_name ):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path,
                                   reporthook=t.update_to)


def dataset_extraction(file_name=file_name,
                       outpath=xoutpath):
    with tar.open(file_name) as file:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(file, outpath)


def download_and_extract(force_download=False,
                         dfile=file_name):
    if force_download:
        dataset_downloader()
    if not os.path.isfile(dfile):
        dataset_downloader()
    print("Start Extracting")
    dataset_extraction()
    print("Extraction Complete")


if __name__ == '__main__':
    download_and_extract()
