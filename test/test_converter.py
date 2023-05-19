from lqcv.data.converter import YOLOConverter, XMLConverter


def test_yolo(label_dir, class_names, img_dir=None):
    converter = YOLOConverter(label_dir, class_names, img_dir)
    print(converter)
    converter.visualize()

def test_xml(label_dir, class_names, img_dir=None):
    converter = XMLConverter(label_dir, class_names, img_dir)
    print(converter)
    converter.visualize()


if __name__ == "__main__":
    # test_yolo(
    #     label_dir="/d/dataset/ultralytics_test/test/suit_mask/labels",
    #     class_names=["mask", "unmask", "suit", "unsuit"],
    #     img_dir=None
    # )
    test_xml(
        label_dir="/d/dataset/ultralytics_test/test/suit_mask/xmls",
        class_names=["mask", "unmask", "suit", "unsuit"],
        img_dir=None
    )
