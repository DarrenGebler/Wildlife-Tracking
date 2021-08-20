from Tracking import GlobalNearestNeighbour
import exiftool
import glob
import os
import datetime
from dateutil.parser import *

folder = "H:/Thesis/Data/2019-08-21_UAS-104_Rockhampton_South/flight3/"


def date_to_int(str_date):
    date = parse(str_date)
    return int(date.strftime('%Y%m%d%H%M%S'))


def get_exif_data():
    files = []
    for file in sorted(glob.glob(folder + "*.jpg")):
        files.append(file)

    with exiftool.ExifTool() as et:
        metadata = et.get_metadata_batch(files)

    data = {}
    for d in metadata:
        filename = d['File:FileName'].split(".jpg")[0]
        data[filename] = {}
        data[filename].update({"width": d['EXIF:ExifImageWidth']})
        data[filename].update({"height": d['EXIF:ExifImageHeight']})
        data[filename].update({"latitude": d['Composite:GPSLatitude']})
        data[filename].update({"longitude": d['Composite:GPSLongitude']})
        data[filename].update({"alt": d['XMP:AbsoluteAltitude']})
        data[filename].update({"roll": d['XMP:GimbalRollDegree']})
        data[filename].update({"yaw": d['XMP:GimbalYawDegree']})
        data[filename].update({"pitch": d['XMP:GimbalPitchDegree']})
        data[filename].update({"time": date_to_int(d['EXIF:CreateDate'])})

    return data


def scale(detection, width, height):
    detection = detection.split(" ")[2:-2]
    return [width * float(detection[0]), height * float(detection[1])]


def get_detection_data(width=640, height=512):
    detection_data = {}
    for file in sorted(glob.glob(folder + "*.txt")):
        with open(file) as f:
            detections = f.read().split("\n")[:-1]
            if len(detections) != 0:
                detections = [scale(detection, width, height) for detection in detections]
            detection_data[os.path.basename(file).split(".txt")[0]] = detections

    return detection_data


if __name__ == '__main__':
    exif_data = get_exif_data()
    det_data = get_detection_data()
    gnn = GlobalNearestNeighbour.GlobalNearestNeighbour()
    for index, key in enumerate(exif_data.keys()):
        if key == "DJI_0283":
            print()
        gnn.update_tracks(det_data[key], exif_data[key]["yaw"], exif_data[key]["pitch"], exif_data[key]["roll"],
                          exif_data[key]["latitude"], exif_data[key]["longitude"], exif_data[key]["alt"],
                          exif_data[key]["time"])
