import os
import pdb
import cv2
from soccer_annotations import get_scaled_annotations_PVOC
from evaluate import read_output_file


class CVAT_annotation(object):
    """CVAT annotation class."""

    def __init__(self):
        """Init."""
        self.content = '<?xml version="1.0" encoding="utf-8"?>\n' +\
                       '<annotations>\n' +\
                       '<version>1.1</version>\n' +\
                       '{}\n' +\
                       '</annotations>'
        self.tracks = []
        self.track_id = 0

    def insert_track(self, bboxes, label="player"):
        """Insert new track."""
        track = '<track id="{}" label="{}">\n' +\
                '{}\n' +\
                '</track>'
        track = track.format(self.track_id, label, "\n".join(bboxes))
        self.track_id += 1
        self.tracks.append(track)

    def create_bbox(self, frame, xtl, ytl, xbr, ybr):
        """Create bbox tag."""
        bbox = '<box frame="{}" outside="0" occluded="0" keyframe="1" ' +\
               'xtl="{}" ytl="{}" xbr="{}" ybr="{}"></box>'
        bbox = bbox.format(frame, xtl, ytl, xbr, ybr)
        return bbox

    def build(self):
        """Generate final file content."""
        xml_file = self.content.format("\n".join(self.tracks))
        return xml_file


def process_annotation_vijay(filename, prefix="", fcode="", output_folder=""):
    """Convert vijay annotations to YOLO train format."""
    include_list = ['Javier_MAscherano', 'Ezequiel_Garay', 'Lionel_Messi', 'Rodrigo_Palacio', 'Fernando_Gago',
                    'Jerome_Boateng', 'Sergio_Aguero', 'Lucas_Biglia', 'Andre_Schurrle', 'Mario_Gotze', 'Thomas_Mueller',
                    'Enzo_Perez', 'Team2', 'Pabli_Zabaleta', 'Mesut_Oezil', 'Marcos_Rojo', 'Phillip_Lahm',
                    'Schweinsteiger', 'Benedikt_Hoewedes', 'Team1', 'Mats_Hummels', 'Christoph_Kramer', 'Referee',
                    'Manuel_Neuer', 'Toni_Kroos', 'Sergio_Romero', 'Martin_Demichelis',
                    'Miroslav_Klose', 'Ezequiel_Lavezzi', 'Gonzalo_Huguain']

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
    """Convert yolo format to x,y cords."""
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
    """Convert x,y to yolo format."""
    xmin, ymin, xmax, ymax = annotation
    w, h = img_size
    x, y = float(xmin + xmax) / 2, float(ymin + ymax) / 2
    x, y = x / w, y / h
    w_rel, h_rel = float(xmax - xmin) / w, float(ymax - ymin) / h
    return [x, y, w_rel, h_rel]


def process_images(image_dir, new_size, output_dir, valid_files=[], fcode=""):
    """Resize images, remove invalid files."""
    images = os.listdir(image_dir)
    for f in images:
        img_path = os.path.join(image_dir, f)
        out_file = os.path.join(output_dir, fcode + "_" + f)
        if len(valid_files) != 0 and out_file not in valid_files:
            continue
        if not f.endswith(".jpg") and not f.endswith(".png"):
            continue

        img = cv2.imread(img_path)
        try:
            img = cv2.resize(img, new_size)
            cv2.imwrite(out_file, img)
        except cv2.error:
            print("[Error] empty:", os.path.join(image_dir, f))


def convert_annotation_xml_to_txt(annotation_folder, output_dir, new_size=(1024, 1024)):
    """Convert PVOC xml to YOLO txt format, image also resizable."""
    files = os.listdir(annotation_folder)
    annotations = get_scaled_annotations_PVOC(annotation_folder, new_size)
    for f in files:
        if f.endswith(".xml"):
            img_name = f.replace(".xml", ".png")
            annotation = annotations[img_name]
            annot = [convert_to_yolo_annot(x, new_size) for x in annotation]
            out = open(os.path.join(output_dir, img_name + ".txt"), "w")
            for a in annot:
                out_str = "0 " + " ".join([str(x) for x in a]) + "\n"
                out.write(out_str)
            out.close()
        print("Processed", f)


def coonvert_to_cvat_format(output_path, output_file, thresh=0.5):
    """Convert pre-detections to CVAT uploadable format."""
    files = [x for x in os.listdir(output_path) if x.endswith(".txt")]
    cvat_annot = CVAT_annotation()
    for j, f in enumerate(files):
        print("Processing file {}/{}".format(j, len(files)))
        annots, conf = read_output_file(os.path.join(output_path, f))
        bboxes = []
        for i, c in enumerate(conf):
            if c >= thresh:
                bbox = cvat_annot.create_bbox(f.strip(".txt"), *annots[i])
                bboxes.append(bbox)
        cvat_annot.insert_track(bboxes)

    with open(output_file, "w") as file:
        file.write(cvat_annot.build())


if __name__ == "__main__":
    # annot_folder = "/home/chris/Downloads/soccer/soccer/annot/"
    # img_folder = "/home/chris/Downloads/soccer/soccer/frames/"
    #
    # annot_out = "/home/chris/Downloads/soccer/soccer/labels/"
    # img_out = "/home/chris/Downloads/soccer/soccer/images/"

    # files = os.listdir(annot_folder)
    # for f in files:
    #     filename = os.path.join(annot_folder, f)
    #     fcode = f.strip(".txt").strip("clip")
    #     print("Processing clip annotations", fcode)
    #     valid_files = process_annotation_vijay(filename, img_folder, fcode, annot_out)
    #     valid_files = [x.replace(annot_out, img_out).replace(".txt", ".jpg") for x in valid_files]
    #     print("Processing clip images", fcode)
    #     process_images(os.path.join(img_folder, fcode), (1024, 1024), img_out, valid_files, fcode)
    #

    # filename = os.path.join(annot_folder, "clip1.txt")
    # fcode = "1"
    # print("Processing clip annotations", fcode)
    # process_annotation(filename, img_folder, fcode, annot_out)
    # print("Processing clip images", fcode)
    # process_images(os.path.join(img_folder, fcode), (1024, 1024), img_out, fcode)
    # pdb.set_trace()

    # annot_folder = "/home/chrizandr/sports/test/annotations"
    # output_folder = "/home/chrizandr/sports/test/labels"
    # convert_annotation_xml_to_txt(annot_folder, output_folder, (1024, 1024))
    output_folder = "/home/chrizandr/yolov3/output/belgium_england_fifa19.mkv-00.01.13.488-00.15.03.905.mkv_out"
    coonvert_to_cvat_format(output_folder, "output.xml")
