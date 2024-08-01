import torch.utils.data as data

import os
import re
import csv
import json
import torch
import tarfile
import pickle
import numpy as np
import pandas as pd
import random
import math
random.seed(2021)
from PIL import Image
from scipy import io as scio
from math import radians, cos, sin, asin, sqrt, pi
IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def get_spatial_info(latitude, longitude):
    if (latitude and longitude) and not (
        math.isnan(latitude) or math.isnan(longitude)
    ):
        latitude = radians(latitude)
        longitude = radians(longitude)
        x = cos(latitude)*cos(longitude)
        y = cos(latitude)*sin(longitude)
        z = sin(latitude)

        if math.isnan(x) or math.isnan(y) or math.isnan(z):
            print("x ", x, " y ", y, " z ", z, " lat ", latitude, " long ",
                  longitude)

        return [x, y, z]
    else:
        return [0, 0, 0]


def get_temporal_info(date, miss_hour=False):
    try:
        if date:
            if miss_hour:
                pattern = re.compile(r'(\d*)-(\d*)-(\d*)', re.I)
            else:
                pattern = re.compile(r'(\d*)-(\d*)-(\d*) (\d*):(\d*):(\d*)', re.I)
            m = pattern.match(date.strip())

            if m:
                year = int(m.group(1))
                month = int(m.group(2))
                day = int(m.group(3))

                if math.isnan(year) or math.isnan(month) or math.isnan(day):
                    print("year ", year, " month ", month, " day ", day)
                    year, month, day = 0, 0, 0

                x_month = sin(2*pi*month/12)
                y_month = cos(2*pi*month/12)

                if x_month.isnan() or y_month.isnan():
                    print("x_month ", x_month, " y_month ", y_month)

                if miss_hour:
                    x_hour = 0
                    y_hour = 0
                else:
                    hour = int(m.group(4))
                    x_hour = sin(2*pi*hour/24)
                    y_hour = cos(2*pi*hour/24)
                return [x_month, y_month, x_hour, y_hour]
            else:
                return [0, 0, 0, 0]
        else:
            return [0, 0, 0, 0]
    except:
        return [0, 0, 0, 0]


def load_file(root,dataset):
    if dataset == 'inaturelist2017':
        year_flag = 7
    elif dataset == 'inaturelist2018':
        year_flag = 8
    
    if dataset == 'inaturelist2018':
        with open(os.path.join(root,'categories.json'),'r') as f:
            map_label = json.load(f)
        map_2018 = dict()
        for _map in map_label:
            map_2018[int(_map['id'])] = _map['name'].strip().lower()
    with open(os.path.join(root,f'val201{year_flag}_locations.json'),'r') as f:
        val_location = json.load(f)
    val_id2meta = dict()
    for meta_info in val_location:
        val_id2meta[meta_info['id']] = meta_info
    with open(os.path.join(root,f'train201{year_flag}_locations.json'),'r') as f:
        train_location = json.load(f)
    train_id2meta = dict()
    for meta_info in train_location:
        train_id2meta[meta_info['id']] = meta_info
    with open(os.path.join(root,f'val201{year_flag}.json'),'r') as f:
        val_class_info = json.load(f)
    with open(os.path.join(root,f'train201{year_flag}.json'),'r') as f:
        train_class_info = json.load(f)
    
    if dataset == 'inaturelist2017':
        categories_2017 = [x['name'].strip().lower() for x in val_class_info['categories']]
        class_to_idx = {c: idx for idx, c in enumerate(categories_2017)}
        id2label = dict()
        for categorie in val_class_info['categories']:
            id2label[int(categorie['id'])] = categorie['name'].strip().lower()
    elif dataset == 'inaturelist2018':
        categories_2018 = [x['name'].strip().lower() for x in map_label]
        class_to_idx = {c: idx for idx, c in enumerate(categories_2018)}
        id2label = dict()
        for categorie in val_class_info['categories']:
            name = map_2018[int(categorie['name'])]
            id2label[int(categorie['id'])] = name.strip().lower()
    
    return train_class_info,train_id2meta,val_class_info,val_id2meta,class_to_idx,id2label
def find_images_and_targets_cub200(root,dataset,istrain=False,aux_info=False):
    imageid2label = {}
    with open(os.path.join(os.path.join(root,'CUB_200_2011'),'image_class_labels.txt'),'r') as f:
        for line in f:
            image_id,label = line.split()
            imageid2label[int(image_id)] = int(label)-1
    imageid2split = {}
    with open(os.path.join(os.path.join(root,'CUB_200_2011'),'train_test_split.txt'),'r') as f:
        for line in f:
            image_id,split = line.split()
            imageid2split[int(image_id)] = int(split)
    images_and_targets = []
    images_info = []
    images_root = os.path.join(os.path.join(root,'CUB_200_2011'),'images')
    bert_embedding_root = os.path.join(root,'bert_embedding_cub')
    text_root = os.path.join(root,'text_c10')
    with open(os.path.join(os.path.join(root,'CUB_200_2011'),'images.txt'),'r') as f:
        for line in f:
            image_id,file_name = line.split()
            file_path = os.path.join(images_root,file_name)
            target = imageid2label[int(image_id)]
            if aux_info:
                with open(os.path.join(bert_embedding_root,file_name.replace('.jpg','.pickle')),'rb') as f_bert:
                    bert_embedding = pickle.load(f_bert)
                    bert_embedding = bert_embedding['embedding_words']
                text_list = []
                with open(os.path.join(text_root,file_name.replace('.jpg','.txt')),'r') as f_text:
                    for line in f_text:
                        line = line.encode(encoding='UTF-8',errors='strict')
                        line = line.replace(b'\xef\xbf\xbd\xef\xbf\xbd',b' ')
                        line = line.decode('UTF-8','strict')
                        text_list.append(line)
            if istrain and imageid2split[int(image_id)]==1:
                if aux_info:
                    images_and_targets.append([file_path,target,bert_embedding])
                    images_info.append({'text_list':text_list})
                else:
                    images_and_targets.append([file_path,target])
            elif not istrain and imageid2split[int(image_id)]==0:
                if aux_info:
                    images_and_targets.append([file_path,target,bert_embedding])
                    images_info.append({'text_list':text_list})
                else:
                    images_and_targets.append([file_path,target])
    return images_and_targets,None,images_info
def find_images_and_targets_cub200_attribute(root,dataset,istrain=False,aux_info=False):
    imageid2label = {}
    with open(os.path.join(os.path.join(root,'CUB_200_2011'),'image_class_labels.txt'),'r') as f:
        for line in f:
            image_id,label = line.split()
            imageid2label[int(image_id)] = int(label)-1
    imageid2split = {}
    with open(os.path.join(os.path.join(root,'CUB_200_2011'),'train_test_split.txt'),'r') as f:
        for line in f:
            image_id,split = line.split()
            imageid2split[int(image_id)] = int(split)
    images_and_targets = []
    images_info = []
    images_root = os.path.join(os.path.join(root,'CUB_200_2011'),'images')
    attributes_root = os.path.join(os.path.join(root,'CUB_200_2011'),'attributes')
    imageid2attribute = {}
    with open(os.path.join(attributes_root,'image_attribute_labels.txt'),'r') as f:
        for line in f:
            if len(line.split())==6:
                image_id,attribute_id,is_present,_,_,_ = line.split()
            else:
                image_id,attribute_id,is_present,certainty_id,time = line.split()
            if int(image_id) not in imageid2attribute:
                imageid2attribute[int(image_id)] = [0 for i in range(312)]
            imageid2attribute[int(image_id)][int(attribute_id)-1] = int(is_present)
    with open(os.path.join(os.path.join(root,'CUB_200_2011'),'images.txt'),'r') as f:
        for line in f:
            image_id,file_name = line.split()
            file_path = os.path.join(images_root,file_name)
            target = imageid2label[int(image_id)]
            if aux_info:
                pass
            if istrain and imageid2split[int(image_id)]==1:
                if aux_info:
                    images_and_targets.append([file_path,target,imageid2attribute[int(image_id)]])
                    images_info.append({'attributes':imageid2attribute[int(image_id)]})
                else:
                    images_and_targets.append([file_path,target])
            elif not istrain and imageid2split[int(image_id)]==0:
                if aux_info:
                    images_and_targets.append([file_path,target,imageid2attribute[int(image_id)]])
                    images_info.append({'attributes':imageid2attribute[int(image_id)]})
                else:
                    images_and_targets.append([file_path,target])
    return images_and_targets,None,images_info
def find_images_and_targets_oxfordflower(root,dataset,istrain=False,aux_info=False):
    imagelabels = scio.loadmat(os.path.join(root,'imagelabels.mat'))
    imagelabels = imagelabels['labels'][0]
    train_val_split = scio.loadmat(os.path.join(root,'setid.mat'))
    train_data = train_val_split['trnid'][0].tolist()
    val_data = train_val_split['valid'][0].tolist()
    test_data = train_val_split['tstid'][0].tolist()
    images_and_targets = []
    images_info = []
    images_root = os.path.join(root,'jpg')
    bert_embedding_root = os.path.join(root,'bert_embedding_flower')
    if istrain:
        all_data = train_data+val_data
    else:
        all_data = test_data
    for data in all_data:
        file_path = os.path.join(images_root,f'image_{str(data).zfill(5)}.jpg')
        target = int(imagelabels[int(data)-1])-1
        if aux_info:
            with open(os.path.join(bert_embedding_root,f'image_{str(data).zfill(5)}.pickle'),'rb') as f_bert:
                bert_embedding = pickle.load(f_bert)
                bert_embedding = bert_embedding['embedding_full']
            images_and_targets.append([file_path,target,bert_embedding])
        else:
            images_and_targets.append([file_path,target])
    return images_and_targets,None,images_info
def find_images_and_targets_stanforddogs(root,dataset,istrain=False,aux_info=False):
    if istrain:
        anno_data = scio.loadmat(os.path.join(root,'train_list.mat'))
    else:
        anno_data = scio.loadmat(os.path.join(root,'test_list.mat'))
    images_and_targets = []
    images_info = []
    for file,label in zip(anno_data['file_list'],anno_data['labels']):
        file_path = os.path.join(os.path.join(root,'Images'),file[0][0])
        target = int(label[0])-1
        images_and_targets.append([file_path,target])
    return images_and_targets,None,images_info
def find_images_and_targets_nabirds(root,dataset,istrain=False,aux_info=False):
    root = os.path.join(root,'nabirds')
    image_paths = pd.read_csv(os.path.join(root,'images.txt'),sep=' ',names=['img_id','filepath'])
    image_class_labels = pd.read_csv(os.path.join(root,'image_class_labels.txt'),sep=' ',names=['img_id','target'])
    label_list = list(set(image_class_labels['target']))
    label_list = sorted(label_list)
    label_map = {k: i for i, k in enumerate(label_list)}
    train_test_split = pd.read_csv(os.path.join(root, 'train_test_split.txt'), sep=' ', names=['img_id', 'is_training_img'])
    data = image_paths.merge(image_class_labels, on='img_id')
    data = data.merge(train_test_split, on='img_id')
    if istrain:
        data = data[data.is_training_img == 1]
    else:
        data = data[data.is_training_img == 0]
    images_and_targets = []
    images_info = []
    for index,row in data.iterrows():
        file_path = os.path.join(os.path.join(root,'images'),row['filepath'])
        target = int(label_map[row['target']])
        images_and_targets.append([file_path,target])
    return images_and_targets,None,images_info
def find_images_and_targets_stanfordcars_v1(root,dataset,istrain=False,aux_info=False):
    if istrain:
        flag = 'train'
    else:
        flag = 'test'
    if istrain:
        anno_data = scio.loadmat(os.path.join(os.path.join(root,'devkit'),f'cars_{flag}_annos.mat'))
    else:
        anno_data = scio.loadmat(os.path.join(os.path.join(root,'devkit'),f'cars_{flag}_annos_withlabels.mat'))
    annotation = anno_data['annotations']
    images_and_targets = []
    images_info = []
    for r in annotation[0]:
        _,_,_,_,label,name = r
        file_path = os.path.join(os.path.join(root,f'cars_{flag}'),name[0])
        target = int(label[0][0])-1
        images_and_targets.append([file_path,target])
    return images_and_targets,None,images_info
def find_images_and_targets_stanfordcars(root,dataset,istrain=False,aux_info=False):
    anno_data = scio.loadmat(os.path.join(root,'cars_annos.mat'))
    annotation = anno_data['annotations']
    images_and_targets = []
    images_info = []
    for r in annotation[0]:
        name,_,_,_,_,label,split = r
        file_path = os.path.join(root,name[0])
        target = int(label[0][0])-1
        if istrain and int(split[0][0])==0:
            images_and_targets.append([file_path,target])
        elif not istrain and int(split[0][0])==1:
            images_and_targets.append([file_path,target])
    return images_and_targets,None,images_info        
def find_images_and_targets_aircraft(root,dataset,istrain=False,aux_info=False):
    file_root = os.path.join(root,'fgvc-aircraft-2013b','data')
    if istrain:
        data_file = os.path.join(file_root,'images_variant_trainval.txt')
    else:
        data_file = os.path.join(file_root,'images_variant_test.txt')
    classes = set()
    with open(data_file,'r') as f:
        for line in f:
            class_name = '_'.join(line.split()[1:])
            classes.add(class_name)
    classes = sorted(list(classes))
    class_to_idx = {name:ind for ind,name in enumerate(classes)}
    
    images_and_targets = []
    images_info = []
    with open(data_file,'r') as f:
        images_root = os.path.join(file_root,'images')
        for line in f:
            image_file = line.split()[0]
            class_name = '_'.join(line.split()[1:])
            file_path = os.path.join(images_root,f'{image_file}.jpg')
            target = class_to_idx[class_name]
            images_and_targets.append([file_path,target])
    return images_and_targets,class_to_idx,images_info
            
def find_images_and_targets_2017_2018(root,dataset,istrain=False,aux_info=False):
    train_class_info,train_id2meta,val_class_info,val_id2meta,class_to_idx,id2label = load_file(root,dataset)
    miss_hour = (dataset == 'inaturelist2017')

    class_info = train_class_info if istrain else val_class_info
    id2meta = train_id2meta if istrain else val_id2meta
    images_and_targets = []
    images_info = []
    if aux_info:
        temporal_info = []
        spatial_info = []
    for image,annotation in zip(class_info['images'],class_info['annotations']):
        file_path = os.path.join(root,image['file_name'])
        id_name = id2label[int(annotation['category_id'])]
        target = class_to_idx[id_name]
        image_id = image['id']
        date = id2meta[image_id]['date']
        latitude = id2meta[image_id]['lat']
        longitude = id2meta[image_id]['lon']
        location_uncertainty = id2meta[image_id]['loc_uncert']
        images_info.append({'date':date,
                'latitude':latitude,
                'longitude':longitude,
                'location_uncertainty':location_uncertainty,
                'target':target}) 
        if aux_info:
            temporal_info = get_temporal_info(date,miss_hour=miss_hour)
            spatial_info = get_spatial_info(latitude,longitude)
            images_and_targets.append((file_path,target,temporal_info+spatial_info))
        else:
            images_and_targets.append((file_path,target))
    return images_and_targets,class_to_idx,images_info
def find_images_and_targets(root,istrain=False,aux_info=False):
    if os.path.exists(os.path.join(root,'train.json')):
        with open(os.path.join(root,'train.json'),'r') as f:
            train_class_info = json.load(f)
    elif os.path.exists(os.path.join(root,'train_mini.json')):
        with open(os.path.join(root,'train_mini.json'),'r') as f:
            train_class_info = json.load(f)
    else:
        raise ValueError(f'not eixst file {root}/train.json or {root}/train_mini.json')
    with open(os.path.join(root,'val.json'),'r') as f:
        val_class_info = json.load(f)
    categories_2021 = [x['name'].strip().lower() for x in val_class_info['categories']]
    class_to_idx = {c: idx for idx, c in enumerate(categories_2021)}
    id2label = dict()
    for categorie in train_class_info['categories']:
        id2label[int(categorie['id'])] = categorie['name'].strip().lower()
    class_info = train_class_info if istrain else val_class_info

    images_and_targets = []
    images_info = []
    if aux_info:
        temporal_info = []
        spatial_info = []

    for image,annotation in zip(class_info['images'],class_info['annotations']):
        file_path = os.path.join(root,image['file_name'])
        id_name = id2label[int(annotation['category_id'])]
        target = class_to_idx[id_name]
        date = image['date']
        latitude = image['latitude']
        longitude = image['longitude']
        location_uncertainty = image['location_uncertainty']
        images_info.append({'date':date,
                'latitude':latitude,
                'longitude':longitude,
                'location_uncertainty':location_uncertainty,
                'target':target}) 
        if aux_info:
            temporal_info = get_temporal_info(date)
            spatial_info = get_spatial_info(latitude,longitude)
            images_and_targets.append((file_path,target,temporal_info+spatial_info))
        else:
            images_and_targets.append((file_path,target))
    return images_and_targets,class_to_idx,images_info


def get_all_classes(*dirs):
    all_classes = set()
    for root_dir in dirs:
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                all_classes.add(class_name)
    return sorted(all_classes)


def find_images_and_targets_inat100k(root, method, train_csv, val_csv, test_csv, nometa=False):
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    test_dir = os.path.join(root, "test")
    main_root = os.path.join(root, method)

    if method == "train":
        metadata_csv = pd.read_csv(train_csv)
    elif method == "val":
        metadata_csv = pd.read_csv(val_csv)
    elif method == "test":
        metadata_csv = pd.read_csv(test_csv)

    # for indexing lat long of csv
    lat_long_cols = ["latitude", "longitude"]
    dates_df = metadata_csv["date_collected"]  # pull out date

    # replace nan lat/longs with median
    lat_long_df = metadata_csv[lat_long_cols].fillna(
        metadata_csv[lat_long_cols].median(0))

    # re-merge dataframes
    metadata_csv = pd.concat([lat_long_df, dates_df], axis=1)

    all_classes = get_all_classes(train_dir, val_dir, test_dir)

    num_classes = len(all_classes)

    # Create a universal class_to_idx mapping
    class_to_idx = {class_name: idx for idx,
                    class_name in enumerate(all_classes)}

    images_and_targets = []
    images_info = []

    aux_info = True  # awesome

    for root, dirs, files in os.walk(main_root, followlinks=True):
        if root == main_root:
            continue

        target_class = root.split("/")[-1]  # parent dir == class
        class_idx = class_to_idx[target_class]  # consistent indicies

        for file in files:
            file_id_str, _ = os.path.splitext(os.path.basename(file))
            file_id = int(file_id_str)

            date = metadata_csv["date_collected"][file_id]
            latitude = metadata_csv['latitude'][file_id]
            longitude = metadata_csv["longitude"][file_id]

            if aux_info:
                if nometa:
                    meta_info = [0, 0, 0, 0] + [0, 0, 0]
                else:
                    # meta_info = get_temporal_info(date, miss_hour=True) + \
                    #             get_spatial_info(latitude, longitude)
                    meta_info = [0, 0, 0, 0] + \
                        get_spatial_info(latitude, longitude)

                images_and_targets.append(
                    (
                        os.path.join(main_root, target_class, file),
                        class_idx,
                        meta_info
                    )
                )

            else:
                images_and_targets.append(
                    (os.path.join(main_root, target_class, file), class_idx)
                )

            images_info.append({
                "date": date,
                "latitude": latitude,
                "longitude": longitude
            })
    return images_and_targets, class_to_idx, images_info, num_classes


def find_images_and_targets_auto_arborist(root, method, train_csv, val_csv, test_csv, nometa=False):
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    test_dir = os.path.join(root, "test")
    main_root = os.path.join(root, method)

    if method == "train":
        metadata_csv = pd.read_csv(train_csv)
    elif method == "val":
        metadata_csv = pd.read_csv(val_csv)
    elif method == "test":
        metadata_csv = pd.read_csv(test_csv)

    # for indexing lat long of csv
    lat_long_cols = ["tree/latitude", "tree/longitude"]
    id_df = metadata_csv["internal_id"]  # pull out date

    # replace nan lat/longs with median
    lat_long_df = metadata_csv[lat_long_cols].fillna(
        metadata_csv[lat_long_cols].median(0))

    # re-merge dataframes
    metadata_csv = pd.concat([lat_long_df, id_df], axis=1)

    all_classes = get_all_classes(train_dir, val_dir, test_dir)

    num_classes = len(all_classes)

    # Create a universal class_to_idx mapping
    class_to_idx = {class_name: idx for idx,
                    class_name in enumerate(all_classes)}

    images_and_targets = []
    images_info = []

    aux_info = True  # awesome

    for root, dirs, files in os.walk(main_root, followlinks=True):
        if root == main_root:
            continue

        target_class = root.split("/")[-1]  # parent dir == class
        class_idx = class_to_idx[target_class]  # consistent indicies

        for file in files:
            file_id, _ = os.path.splitext(os.path.basename(file))

            latitude = metadata_csv.loc[
                metadata_csv['internal_id'] == file_id,
                'tree/latitude'
            ].iloc[0]
            longitude = metadata_csv.loc[
                metadata_csv['internal_id'] == file_id,
                'tree/longitude'
            ].iloc[0]

            if aux_info:
                if nometa:
                    meta_info = [0, 0, 0, 0] + [0, 0, 0]
                else:
                    meta_info = [0, 0, 0, 0] + \
                      get_spatial_info(latitude, longitude)

                images_and_targets.append(
                    (
                        os.path.join(main_root, target_class, file),
                        class_idx,
                        meta_info
                    )
                )

            else:
                images_and_targets.append(
                    (os.path.join(main_root, target_class, file), class_idx)
                )

            images_info.append({
                "latitude": latitude,
                "longitude": longitude
            })
    return images_and_targets, class_to_idx, images_info, num_classes


class DatasetMeta(data.Dataset):
    def __init__(
            self,
            root,
            load_bytes=False,
            transform=None,
            train=False,
            aux_info=False,
            dataset='inaturelist2021',
            class_ratio=1.0,
            per_sample=1.0,
            train_csv="",
            val_csv="",
            test_csv="",
            nometa=False,
            istest=False
    ):
        self.aux_info = aux_info
        self.dataset = dataset
        if dataset in ['inaturelist2021','inaturelist2021_mini']:
            images, class_to_idx, images_info = find_images_and_targets(root,train,aux_info)
        elif dataset in ['inaturelist2017','inaturelist2018']:
            images, class_to_idx, images_info = find_images_and_targets_2017_2018(root,dataset,train,aux_info)
        elif dataset == 'cub-200':
            images, class_to_idx,images_info = find_images_and_targets_cub200(root,dataset,train,aux_info)
        elif dataset == 'stanfordcars':
            images, class_to_idx,images_info = find_images_and_targets_stanfordcars(root,dataset,train)
        elif dataset == 'oxfordflower':
            images, class_to_idx,images_info = find_images_and_targets_oxfordflower(root,dataset,train,aux_info)
        elif dataset == 'stanforddogs':
            images,class_to_idx,images_info = find_images_and_targets_stanforddogs(root,dataset,train)
        elif dataset == 'nabirds':
            images,class_to_idx,images_info = find_images_and_targets_nabirds(root,dataset,train)
        elif dataset == 'aircraft':
            images,class_to_idx,images_info = find_images_and_targets_aircraft(root,dataset,train)

        elif dataset == "inat100k":
            path = "train" if train else "val"
            if istest:
                path = "test"

            (
                images,
                class_to_idx,
                images_info,
                _
            ) = find_images_and_targets_inat100k(
                root,
                path,
                train_csv,
                val_csv,
                test_csv,
                nometa=nometa
            )
        elif dataset == "auto_arborist":
            path = "train" if train else "val"
            if istest:
                path = "test"
            (
                images,
                class_to_idx,
                images_info,
                num_classes
            ) = find_images_and_targets_auto_arborist(
                root,
                path,
                train_csv,
                val_csv,
                test_csv,
                nometa=nometa
            )
            self.num_classes = num_classes  # DANGEROUS

        if len(images) == 0:
            raise RuntimeError(
                f'Found 0 images in subfolders of {root}. '
                f'Supported image extensions are {", ".join(IMG_EXTENSIONS)}')
        self.root = root
        self.samples = images
        self.imgs = self.samples  # torchvision ImageFolder compat
        self.class_to_idx = class_to_idx
        self.images_info = images_info
        self.load_bytes = load_bytes
        self.transform = transform

    def __getitem__(self, index):
        if self.aux_info:
            path, target, aux_info = self.samples[index]
        else:
            path, target = self.samples[index]
        img = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.aux_info:
            if type(aux_info) is np.ndarray:
                select_index = np.random.randint(aux_info.shape[0])
                return img, target, aux_info[select_index,:]
            else:
                return img, target, np.asarray(aux_info).astype(np.float64)
        else:
            return img, target

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
#     train_dataset = DatasetPre('./fgvc_previous','./fgvc_previous',train=True,aux_info=True)
#     import ipdb;ipdb.set_trace()
#     train_dataset = DatasetMeta('./nabirds',train=True,aux_info=False,dataset='nabirds')
#     find_images_and_targets_stanforddogs('./stanforddogs',None,istrain=True)
#     find_images_and_targets_oxfordflower('./oxfordflower',None,istrain=True)
    find_images_and_targets_ablation('./inaturelist2021',True,True,0.5,1.0)
#     find_images_and_targets_cub200('./cub-200','cub-200',True,True)
#     find_images_and_targets_aircraft('./aircraft','aircraft',True)
#     train_dataset = DatasetMeta('./aircraft',train=False,aux_info=False,dataset='aircraft')
    import ipdb;ipdb.set_trace()
#     find_images_and_targets_2017('')
