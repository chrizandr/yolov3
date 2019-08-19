import os
import pdb
import cv2


include_list = ['Javier_MAscherano', 'Ezequiel_Garay', 'Lionel_Messi', 'Rodrigo_Palacio', 'Fernando_Gago',
                'Jerome_Boateng', 'Sergio_Aguero', 'Lucas_Biglia', 'Andre_Schurrle', 'Mario_Gotze', 'Thomas_Mueller',
                'Enzo_Perez', 'Team2', 'Pabli_Zabaleta', 'Mesut_Oezil', 'Marcos_Rojo', 'Phillip_Lahm',
                'Schweinsteiger', 'Benedikt_Hoewedes', 'Team1', 'Mats_Hummels', 'Christoph_Kramer', 'Referee',
                'Manuel_Neuer', 'Toni_Kroos', 'Sergio_Romero', 'Martin_Demichelis',
                'Miroslav_Klose', 'Ezequiel_Lavezzi', 'Gonzalo_Huguain']


def process_annotation(filename, prefix="", fcode="", output_folder=""):
    f = open(filename, "r")
    annotations = {}

    for line in f:
        data = line.split()
        annot = [int(x) for x in data[1:5]]
        annot = convert_to_yolo_annot(annot, img_size=(720, 402))
        img_key = os.path.join(output_folder, fcode + "_" + data[5] + ".txt")
        lost, occluded, generated = [int(x) for x in data[6:9]]
        label = data[9].strip('"').strip("'")
        if lost == 0 and label in include_list:
            if img_key in annotations:
                annotations[img_key].append(annot)
            else:
                annotations[img_key] = [annot, ]

    for f in annotations:
        with open(f, "a+") as file:
            for x in annotations[f]:
                s = ["0"] + [str(y) for y in x]
                file.write(" ".join(s) + "\n")
    return list(annotations.keys())


def convert_from_yolo_annot(annotation, img_size=(720, 402)):
    x, y, w, h = annotation
    w_img, h_img = img_size
    x_max_p_min = x * w_img * 2
    y_max_p_min = y * h_img * 2
    x_max_m_min = w * w_img
    y_max_m_min = h * h_img
    xmax = int((x_max_p_min + x_max_m_min) / 2)
    ymax = int((y_max_p_min + y_max_m_min) / 2)
    xmin = int((x_max_p_min - x_max_m_min) / 2)
    ymin = int((y_max_p_min - y_max_m_min) / 2)
    return [xmin, ymin, xmax, ymax]


def convert_to_yolo_annot(annotation, img_size=(720, 402)):
    xmin, ymin, xmax, ymax = annotation
    w, h = img_size
    x, y = float(xmin + xmax) / 2, float(ymin + ymax) / 2
    x, y = x / w, y / h
    w_rel, h_rel = float(xmax - xmin) / w, float(ymax - ymin) / h
    return [x, y, w_rel, h_rel]


def process_images(image_dir, new_size, output_dir, valid_files=[], fcode=""):
    images = os.listdir(image_dir)
    for f in images:
        img_path = os.path.join(image_dir, f)
        out_file = os.path.join(output_dir, fcode + "_" + f)
        if len(valid_files) != 0 and out_file not in valid_files:
            continue
        if not f.endswith(".jpg"):
            continue

        img = cv2.imread(img_path)
        try:
            img = cv2.resize(img, new_size)
            cv2.imwrite(out_file, img)
        except cv2.error:
            print("[Error] empty:", os.path.join(image_dir, f))


if __name__ == "__main__":
    annot_folder = "/home/chris/Downloads/soccer/soccer/annot/"
    img_folder = "/home/chris/Downloads/soccer/soccer/frames/"

    annot_out = "/home/chris/Downloads/soccer/soccer/labels/"
    img_out = "/home/chris/Downloads/soccer/soccer/images/"

    files = os.listdir(annot_folder)
    for f in files:
        filename = os.path.join(annot_folder, f)
        fcode = f.strip(".txt").strip("clip")
        print("Processing clip annotations", fcode)
        valid_files = process_annotation(filename, img_folder, fcode, annot_out)
        valid_files = [x.replace(annot_out, img_out).replace(".txt", ".jpg") for x in valid_files]
        print("Processing clip images", fcode)
        process_images(os.path.join(img_folder, fcode), (1024, 1024), img_out, valid_files, fcode)

    # filename = os.path.join(annot_folder, "clip1.txt")
    # fcode = "1"
    # print("Processing clip annotations", fcode)
    # process_annotation(filename, img_folder, fcode, annot_out)
    # print("Processing clip images", fcode)
    # process_images(os.path.join(img_folder, fcode), (1024, 1024), img_out, fcode)
    # pdb.set_trace()