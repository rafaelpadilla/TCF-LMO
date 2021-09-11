import fnmatch
import json
import os

import cv2
import torch

from .definitions import (
    PATH_JSON_FOLDS,
    JSON_file,
    Type_dataset,
    csv_dir,
    target_objects,
    videos_dir,
)


def unnormalize(img, transf_std, transf_mean, one_channel=False):
    if one_channel:
        transf_std = [transf_std]
        transf_mean = [transf_mean]
    # unnormalize
    img = (img * torch.FloatTensor(transf_std).unsqueeze(1).unsqueeze(1).to(
        img.device)) + torch.FloatTensor(transf_mean).unsqueeze(1).unsqueeze(1).to(img.device)
    return img


def get_video_files_from_target_url(url_video):
    # Dada a url de um vídeo alvo 'http://www02.smt.ufrj.br/~tvdigital/database/objects/data/avi/obj-sing-amb-part03-video12.avi'
    # retorna o caminho local do vídeo alvo associado, de referência e o arquivo de anotação (alignment)
    # Não pode ser url de um vídeo de referência ex: 'http://www02.smt.ufrj.br/~tvdigital/database/objects/data/001/avi/ref-sing-ext-part03-video01.avi'
    assert '/ref-' not in url_video, 'The video url must be a target video. The given url is a reference video'
    # Define dicionário com os caminhos locais para os vídeos ref e alvo e arquivo de anotação
    ret = {'alignment': '', 'reference': '', 'target': ''}

    # É um vídeo de treino
    if '/database/objects/' in url_video:
        ds_type = Type_dataset.Train
        dir_ds = videos_dir['train']
    # É um vídeo de validação ou de teste
    elif '/database/research/' in url_video:
        ds_type = Type_dataset.Validation  # ou poderia ser Type_dataset.Test
        dir_ds = videos_dir[
            'validation']  # ou poderia set videos_dir['test'] (diretório com vídeos é o mesmo)
    # Chama as funções para obter os caminhos dos vídeos na máquina local
    ret['reference'] = get_path_reference_video(url_video, ds_type)[0]
    ret['target'] = find_file_in_dir_recursively(dir_ds, os.path.basename(url_video))[0]
    ret['alignment'] = get_alignment_file(ret['target'], ds_type)
    return ret


def get_fold(fold_number):
    assert fold_number >= 1 and fold_number <= 72, 'Fold deve ser representado por um número entre 1 e 72.'
    folds = get_all_folds()[str(fold_number)]
    for db, objs in folds['objects'].items():
        if not isinstance(objs, list):
            folds['objects'][db] = [objs]
    return folds


def get_file_paths_from_object_info(dict_obj_info, dataset_type):
    if dataset_type == Type_dataset.Train:
        dir_ds = videos_dir['train']
    elif dataset_type == Type_dataset.Validation:
        dir_ds = videos_dir['validation']
    elif dataset_type == Type_dataset.Test:
        dir_ds = videos_dir['test']
    paths = []
    for table_name, v_table in dict_obj_info.items():
        for o in v_table:
            path_video = find_file_in_dir_recursively(dir_ds, os.path.basename(o['url']))
            assert len(path_video) == 1
            path_video = path_video[0]
            # Garante que o nome da tabela está no caminho do vídeo
            assert table_name in path_video
            paths.append(path_video)
    return paths


def get_all_folds():
    # Se arquivo json com todos fols já tiver sido criado, retorná-lo
    if os.path.isfile(PATH_JSON_FOLDS):
        with open(PATH_JSON_FOLDS, "r") as read_file:
            print(f'Abrindo arquivo {PATH_JSON_FOLDS}.')
            all_folds = json.load(read_file)
            return all_folds

    all_folds = {}
    for tar_obj in target_objects:
        # Descarta objeto alvo
        temp_validation_objs = target_objects.copy()
        temp_validation_objs.remove(tar_obj)
        # Para cada possível objeto de validação, são criados os folds
        for o in temp_validation_objs:
            validation_obj = o
            training_objs = target_objects.copy()
            training_objs.remove(tar_obj)
            training_objs.remove(o)
            # Adiciona grupo de objetos no fold (fold 1, fold 2, ..., fold 72)
            new_fold_number = str(len(all_folds) + 1)
            all_folds[new_fold_number] = {
                'objects': {
                    'test': tar_obj,
                    'train': training_objs,
                    'validation': validation_obj
                }
            }
            ####################################
            # Informações dos vídeos de treino #
            ####################################
            info_vids_train = []
            # Caminhos de vídeos de treino
            train_objs_info = get_objects_info(training_objs, Type_dataset.Train)
            # get_reference_video
            paths_videos_train = get_file_paths_from_object_info(train_objs_info,
                                                                 Type_dataset.Train)
            for path in paths_videos_train:
                # A table 01 tem 2 videos de anotação. Por isso pegamos sempre o primeiro video de anotação
                path_ref_video = get_path_reference_video(path, Type_dataset.Train)[0]
                path_annotation_file = get_path_annotation_file(path, Type_dataset.Train)
                paths_alignment_files = get_alignment_file(path, Type_dataset.Train)
                info_vids_train.append({
                    'reference': path_ref_video,
                    'target': path,
                    'annotation': path_annotation_file,
                    'alignment': paths_alignment_files
                })
            #######################################
            # Informações dos vídeos de validacao #
            #######################################
            info_vids_validation = []
            # Caminhos de vídeos de validacao
            val_objs_info = get_objects_info([validation_obj], Type_dataset.Validation)
            # get_reference_video
            paths_videos_val = get_file_paths_from_object_info(val_objs_info,
                                                               Type_dataset.Validation)
            for path in paths_videos_val:
                # A table 01 tem 2 videos de anotação. Por isso pegamos sempre o primeiro video de anotação
                path_ref_video = get_path_reference_video(path, Type_dataset.Validation)[0]
                path_annotation_file = get_path_annotation_file(path, Type_dataset.Validation)
                paths_alignment_files = get_alignment_file(path, Type_dataset.Validation)
                info_vids_validation.append({
                    'reference': path_ref_video,
                    'target': path,
                    'annotation': path_annotation_file,
                    'alignment': paths_alignment_files
                })
            ###################################
            # Informações dos vídeos de teste #
            ###################################
            info_vids_teste = []
            # Caminhos de vídeos de teste
            tar_objs_info = get_objects_info([tar_obj], Type_dataset.Test)
            # get_reference_video
            paths_videos_test = get_file_paths_from_object_info(tar_objs_info, Type_dataset.Test)
            for path in paths_videos_test:
                # A table 01 tem 2 videos de anotação. Por isso pegamos sempre o primeiro video de anotação
                path_ref_video = get_path_reference_video(path, Type_dataset.Test)[0]
                path_annotation_file = get_path_annotation_file(path, Type_dataset.Test)
                paths_alignment_files = get_alignment_file(path, Type_dataset.Test)
                info_vids_teste.append({
                    'reference': path_ref_video,
                    'target': path,
                    'annotation': path_annotation_file,
                    'alignment': paths_alignment_files
                })
            # Coloca em um dicionário
            all_folds[new_fold_number]['videos_paths'] = {
                'test': info_vids_teste,
                'train': info_vids_train,
                'validation': info_vids_validation
            }
    # Salvar Json para futuramente ser mais eficiente a obtenção de todos os folds
    with open(PATH_JSON_FOLDS, 'w') as json_file:
        json.dump(all_folds, json_file)
        print(f'Arquivo {PATH_JSON_FOLDS} salvo com sucesso.')
    return all_folds


def get_objects_info(classes, dataset_type):
    """ Get a list of objects info.

    Arguments:
        classes {list[strings]} -- list of object classes (eg ['shoe', 'towel', 'brown box', 'black
        coat'])
        Type_dataset {enum} -- Type_dataset enum representing which dataset to use (object or
        research)

    Returns:
        ret {dictionary} -- dictionary containing structure with tables, object, object_class,
        position, url of videos containing the given classes
    """
    dataset_type_str = dataset_type.name
    if dataset_type_str == Type_dataset.Train.name:
        path_json = JSON_file.Object.value
    elif dataset_type_str == Type_dataset.Validation.name or dataset_type_str == Type_dataset.Test.name:
        path_json = JSON_file.Research.value
    else:
        raise Exception('dataset_type must be either Type_dataset.Train or Type_dataset.Test')

    assert os.path.isfile(path_json)

    ret = {}
    with open(path_json, "r") as read_file:
        data = json.load(read_file)
        for t in data['tables']:
            for o in data['tables'][t]['objects']:
                for _class in classes:
                    if o['object_class'] == _class:
                        # if o['object_class'] in classes:
                        if t not in ret:
                            ret[t] = []
                        ret[t].append(o)
    return ret


def get_path_reference_video(video_target, dataset_type):
    video_target = os.path.basename(video_target)
    dataset_type_str = dataset_type.name
    if dataset_type_str == Type_dataset.Train.name:
        path_json = JSON_file.Object.value
        dir_ds = videos_dir['train']
    elif dataset_type_str == Type_dataset.Validation.name:
        dir_ds = videos_dir['validation']
        path_json = JSON_file.Research.value
    elif dataset_type_str == Type_dataset.Test.name:
        dir_ds = videos_dir['test']
        path_json = JSON_file.Research.value
    else:
        raise Exception('dataset_type must be either Type_dataset.Train or Type_dataset.Test')
    assert os.path.isfile(path_json)
    ret = []
    with open(path_json, "r") as read_file:
        data = json.load(read_file)
        for table_name in data['tables']:
            for obj in data['tables'][table_name]['objects']:
                if video_target in obj['url']:
                    for ref in data['tables'][table_name]['references']:
                        path_video = find_file_in_dir_recursively(dir_ds,
                                                                  os.path.basename(ref['url']))
                        assert len(path_video) == 1, 'Arquivo de referência não encontrado.'
                        ret.append(path_video[0])
    return ret


def get_path_annotation_file(video_target, dataset_type):
    video_target = os.path.basename(video_target)
    dataset_type_str = dataset_type.name
    if dataset_type_str == Type_dataset.Train.name:
        path_json = JSON_file.Object.value
        dir_ds = videos_dir['train']
    elif dataset_type_str == Type_dataset.Validation.name:
        dir_ds = videos_dir['validation']
        path_json = JSON_file.Research.value
    elif dataset_type_str == Type_dataset.Test.name:
        dir_ds = videos_dir['test']
        path_json = JSON_file.Research.value
    else:
        raise Exception('dataset_type must be either Type_dataset.Train or Type_dataset.Test')
    assert os.path.isfile(path_json)
    ret = []
    with open(path_json, "r") as read_file:
        data = json.load(read_file)
        for table_name in data['tables']:
            for obj in data['tables'][table_name]['objects']:
                if video_target in obj['url']:
                    if 'url_annotation' not in obj:
                        path_annotation = find_file_in_dir_recursively(
                            dir_ds,
                            os.path.basename(obj['url']).replace('.avi', '.txt'))
                    else:
                        path_annotation = find_file_in_dir_recursively(
                            dir_ds, os.path.basename(obj['url_annotation']))
                    assert len(path_annotation) == 1
                    # Como só existe um arquivo de anotação por video_target, quando encontrou um, retorna, pois ele é único
                    return path_annotation[0]
        return None


def get_alignment_file(video_target, dataset_type):
    video_target_dir = os.path.dirname(video_target)
    dataset_type_str = dataset_type.name
    if dataset_type_str == Type_dataset.Train.name:
        dir_alignment = csv_dir['train']
    elif dataset_type_str == Type_dataset.Validation.name:
        dir_alignment = csv_dir['validation']
    elif dataset_type_str == Type_dataset.Test.name:
        dir_alignment = csv_dir['test']
        path_jon = JSON_file.Research.value
    else:
        raise Exception('dataset_type must be either Type_dataset.Train or Type_dataset.Test')
    assert os.path.isdir(dir_alignment), f'Alignment file {dir_alignment} not found.'
    # Ex de video_targat_dir: '/media/storage/VDAO/vdao_object/table_01/Table_01-Object_02'
    # Obtém as 2 últimas pastas ('table_01/Table_01-Object_02')
    video_target_dir = os.path.normpath(video_target_dir)
    ending_folders = video_target_dir.split(os.sep)[-2:]
    dir_alignment = os.path.join(dir_alignment, ending_folders[0], ending_folders[1])
    assert os.path.isdir(
        dir_alignment
    ), f'Mounted path {dir_alignment} is not a valid directory with annotation files.'
    return get_files_recursively(dir_alignment, '*.csv')



def get_files(directory, extension="*", recursive=True):
    if '.' not in extension:
        extension = '*.' + extension
    if recursive:
        files_ret = [
            os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(directory)
            for f in fnmatch.filter(files, extension)
        ]
    else:
        files_ret = []
        for dirpath, dirnames, files in os.walk(directory):
            for f in fnmatch.filter(files, extension):
                files_ret.append(os.path.join(dirpath, f))
            break

    # Disconsider hidden files, such as .DS_Store in the MAC OS
    ret = [f for f in files_ret if not os.path.basename(f).startswith('.')]
    return ret


def find_file_in_dir_recursively(directory, file_name):
    paths = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        if file_name in filenames:
            paths.append(os.path.join(dirpath, file_name))
    return paths


def get_files_recursively(directory, extension="*"):
    if '.' not in extension:
        extension = '*.' + extension
    files = [
        os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(directory)
        for f in fnmatch.filter(files, extension)
    ]
    return files


def add_bb_into_image(image, boundingBoxes, color, thickness, label=None):
    r = int(color[0])
    g = int(color[1])
    b = int(color[2])

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontThickness = 1
    safetyPixels = 0

    for annotations in boundingBoxes:
        boundingBox = annotations[1]
        label = annotations[0]
        xIn = boundingBox[0]
        yIn = boundingBox[1]
        cv2.rectangle(image, (boundingBox[0], boundingBox[1]), (boundingBox[2], boundingBox[3]),
                      (r, g, b), thickness)
        label = None
        # Add label
        if label != None:
            # Get size of the text box
            (tw, th) = cv2.getTextSize(label, font, fontScale, fontThickness)[0]
            # Top-left coord of the textbox
            (xin_bb, yin_bb) = (xIn + thickness, yIn - th + int(12.5 * fontScale))
            # Checking position of the text top-left (outside or inside the bb)
            if yin_bb - th <= 0:  # if outside the image
                yin_bb = yIn + th  # put it inside the bb
            r_Xin = xIn - int(thickness / 2)
            r_Yin = yin_bb - th - int(thickness / 2)
            # Draw filled rectangle to put the text in it
            cv2.rectangle(image, (r_Xin, r_Yin - thickness),
                          (r_Xin + tw + thickness * 3, r_Yin + th + int(12.5 * fontScale)),
                          (r, g, b), -1)
            cv2.putText(image, label, (xin_bb, yin_bb), font, fontScale, (255, 255, 255),
                        fontThickness, cv2.LINE_AA)
