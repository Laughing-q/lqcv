from lqcv.data.converter import YOLOConverter, XMLConverter, COCOConverter


def test_yolo(label_dir, class_names, img_dir=None):
    converter = YOLOConverter(label_dir, class_names, img_dir)
    print(converter)
    converter.visualize()


def test_xml(label_dir, class_names, img_dir=None):
    converter = XMLConverter(label_dir, class_names, img_dir)
    print(converter)
    converter.visualize()


def test_coco(json_file, img_dir=None):
    converter = COCOConverter(json_file, img_dir)
    print(converter)
    converter.visualize()


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
        json_file="/d/dataset/ultralytics_test/sub_mask_sub/labels/train.json",
        img_dir="/d/dataset/ultralytics_test/sub_mask_sub/images/train",
    )
