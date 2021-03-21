import os
from google_drive_downloader import GoogleDriveDownloader as gdd

#Download training file
gdd.download_file_from_google_drive(file_id='1MkeJlbzueNJdjatyvZ4B4cO4WEnn1MwS',
                                    dest_path='./Data/train.zip',
                                    showsize=True,
                                    unzip=True)

os.remove("./Data/train.zip")