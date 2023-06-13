from lqcv.data.converter import YOLOConverter, XMLConverter, COCOConverter

def _test_converter(converter):
    # to COCO
    converter.toCOCO(save_file="label.json", classes=converter.class_names)
    coco = COCOConverter("label.json", converter.img_dir)
    print(coco)
    coco.visualize()

    # to YOLO
    converter.toYOLO(save_dir="./labels", classes=converter.class_names)
    yolo = YOLOConverter("./labels", class_names=converter.class_names, img_dir=converter.img_dir)
    print(yolo)
    yolo.visualize()

    # to XML
    converter.toXML(save_dir="./xmls", classes=converter.class_names)
    xml = XMLConverter("./xmls", img_dir=converter.img_dir)
    print(xml)
    xml.visualize()


def test_yolo(label_dir, class_names, img_dir=None):
    converter = YOLOConverter(label_dir, class_names, img_dir)
    print(converter)
    # converter.visualize()
    _test_converter(converter)


def test_xml(label_dir, class_names, img_dir=None):
    converter = XMLConverter(label_dir, class_names, img_dir)
    print(converter)
    # converter.visualize()
    _test_converter(converter)


def test_coco(json_file, img_dir=None):
    converter = COCOConverter(json_file, img_dir)
    print(converter)
    # converter.visualize()
    _test_converter(converter)


if __name__ == "__main__":
    # test_yolo(
    #     label_dir="/d/dataset/ultralytics_test/test/suit_mask/labels",
    #     class_names=["mask", "unmask", "suit", "unsuit"],
    #     img_dir=None
    # )

    # test_xml(
    #     label_dir="/d/dataset/ultralytics_test/test/suit_mask/xmls",
    #     # class_names=["mask", "unmask", "suit", "unsuit"],
    #     class_names=None,
    #     img_dir=None
    # )

    test_coco(
        json_file="/d/dataset/COCO/annotations/instances_val2017.json",
        img_dir="/d/dataset/COCO/images/val2017",
    )

    # official coco dataset
    # test_coco(
    #     json_file="/d/dataset/COCO/annotations/instances_val2017.json",
    #     img_dir="/d/dataset/COCO/images/val2017",
    # )
