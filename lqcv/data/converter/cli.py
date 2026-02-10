"""Command-line interface for dataset converters."""

from __future__ import annotations

from pathlib import Path
import click


def get_converter(input_format: str, source: str, class_names: list[str] | None = None, img_dir: str | None = None):
    """Get the appropriate converter based on input format.

    Args:
        input_format: One of 'yolo', 'xml', 'coco'
        source: Path to labels directory or JSON file
        class_names: List of class names (optional for xml/coco)
        img_dir: Path to images directory (optional)

    Returns:
        Converter instance
    """
    if input_format == "yolo":
        from .yolo import YOLOConverter

        return YOLOConverter(source, class_names, img_dir=img_dir)
    elif input_format == "xml":
        from .xml import XMLConverter

        return XMLConverter(source, class_names, img_dir=img_dir)
    elif input_format == "coco":
        from .coco import COCOConverter

        return COCOConverter(source, img_dir=img_dir)
    else:
        raise ValueError(f"Unsupported input format: {input_format}")


@click.group()
def main():
    """LQCV Dataset Converter - Convert between YOLO, XML, and COCO formats."""
    pass


@main.command("convert")
@click.option("--source", required=True, help="Path to labels directory (YOLO/XML) or JSON file (COCO)")
@click.option("--from", "input_format", required=True, type=click.Choice(["yolo", "xml", "coco"]), help="Input format")
@click.option("--to", "output_format", required=True, type=click.Choice(["yolo", "xml", "coco"]), help="Output format")
@click.option("--save-dir", required=True, type=click.Path(), help="Output directory for converted labels")
@click.option("--img-dir", type=click.Path(), help="Images directory (auto-detected if not provided)")
@click.option("--classes", help="Comma-separated class names (e.g., 'person,car,dog')")
@click.option("--classes-file", type=click.Path(exists=True), help="File with class names (one per line)")
@click.option("--filter-classes", help="Comma-separated classes to keep (filters out others)")
@click.option("--single-cls", is_flag=True, default=False, help="Treat all classes as one class (YOLO output only)")
@click.option("--copy-images", type=click.Path(), help="Copy filtered images to this directory")
def convert_command(
    source: str,
    input_format: str,
    output_format: str,
    save_dir: str,
    img_dir: str | None,
    classes: str | None,
    classes_file: str | None,
    filter_classes: str | None,
    single_cls: bool,
    copy_images: str | None,
):
    """Convert dataset between different annotation formats.

    Examples:
        # YOLO to COCO
        lqcv convert --source labels/ --from yolo --to coco --save-dir output.json --classes "person,car,dog"

        # XML to YOLO with filtering
        lqcv convert --source xmls/ --from xml --to yolo --save-dir labels/ --filter-classes "person,car"

        # COCO to XML
        lqcv convert --source annotations.json --from coco --to xml --save-dir xmls/
    """
    # Parse class names
    class_names = None
    if classes:
        class_names = [c.strip() for c in classes.split(",")]
    elif classes_file:
        with open(classes_file) as f:
            class_names = [line.strip() for line in f if line.strip()]

    # Load converter
    click.echo(f"Loading {input_format.upper()} dataset from {source}...")
    converter = get_converter(input_format, source, class_names, img_dir)

    click.echo(f"Dataset loaded: {len(converter.labels)} images")
    # click.echo(converter)

    # Parse filter classes
    filter_class_list = None
    if filter_classes:
        filter_class_list = [c.strip() for c in filter_classes.split(",")]
        click.echo(f"Filtering classes: {filter_class_list}")
    if class_names is None and filter_class_list is not None:  # special handle for yolo format
        try:
            filter_class_list = [int(c) for c in filter_class_list]
        except ValueError:
            click.echo(
                "Error: When class names are not provided in YOLO format, filter classes must be indices (integers)."
            )

    # Convert
    click.echo(f"Converting to {output_format.upper()}...")
    if output_format == "yolo":
        converter.toYOLO(save_dir, classes=filter_class_list, im_dir=copy_images, single_cls=single_cls)
    elif output_format == "xml":
        converter.toXML(save_dir, classes=filter_class_list, im_dir=copy_images)
    elif output_format == "coco":
        converter.toCOCO(save_file=Path(save_dir) / "labels.json", classes=filter_class_list, im_dir=copy_images)

    click.echo(f"✅ Conversion complete! Results saved to {save_dir}")


@main.command("check")
@click.option("--source", required=True, help="Path to labels directory (YOLO/XML) or JSON file (COCO)")
@click.option(
    "--format", "input_format", required=True, type=click.Choice(["yolo", "xml", "coco"]), help="Input format"
)
@click.option("--img-dir", type=click.Path(), help="Images directory (auto-detected if not provided)")
@click.option("--classes", help="Comma-separated class names")
@click.option("--classes-file", type=click.Path(exists=True), help="File with class names (one per line)")
@click.option("--iou-thresh", default=0.7, type=float, help="IoU threshold for overlap detection")
@click.option("--min-pixel", default=5, type=int, help="Minimum width/height in pixels")
@click.option("--filter", "do_filter", is_flag=True, default=False, help="Filter out invalid boxes")
@click.option("--visualize", type=click.Path(), help="Save visualizations of issues to this directory")
def check_command(
    source: str,
    input_format: str,
    img_dir: str | None,
    classes: str | None,
    classes_file: str | None,
    iou_thresh: float,
    min_pixel: int,
    do_filter: bool,
    visualize: str | None,
):
    """Check dataset for common issues (overlaps, tiny boxes, etc.).

    Examples:
        # Check YOLO dataset
        lqcv check --source labels/ --format yolo --classes "person,car,dog"

        # Check and filter issues
        lqcv check --source labels/ --format yolo --filter --classes-file classes.txt

        # Check with visualization
        lqcv check --source xmls/ --format xml --visualize issues/
    """
    # Parse class names
    class_names = None
    if classes:
        class_names = [c.strip() for c in classes.split(",")]
    elif classes_file:
        with open(classes_file) as f:
            class_names = [line.strip() for line in f if line.strip()]

    # Load converter
    click.echo(f"Loading {input_format.upper()} dataset from {source}...")
    converter = get_converter(input_format, source, class_names, img_dir)

    click.echo(f"Dataset loaded: {len(converter.labels)} images")
    click.echo(converter)

    # Check dataset
    click.echo(f"\nChecking dataset (IoU threshold: {iou_thresh}, min pixels: {min_pixel})...")
    converter.check(iou_thres=iou_thresh, min_pixel=min_pixel, filter=do_filter)

    # Report results
    if converter.check_results["iou"]:
        click.echo("\n⚠️  Overlapping boxes found:")
        for (cls1, cls2), count in converter.check_results["iou"].items():
            click.echo(f"  - {cls1} ↔ {cls2}: {count} pairs")

    if do_filter:
        click.echo(f"\n✅ Filtering applied. Invalid boxes removed.")

    # Visualize if requested
    if visualize:
        click.echo(f"\nSaving visualizations to {visualize}...")
        converter.visualize(save_dir=visualize, sign_only=True)
        click.echo(f"✅ Visualizations saved!")


@main.command("visualize")
@click.option("--source", required=True, help="Path to labels directory (YOLO/XML) or JSON file (COCO)")
@click.option(
    "--format", "input_format", required=True, type=click.Choice(["yolo", "xml", "coco"]), help="Input format"
)
@click.option("--img-dir", type=click.Path(), help="Images directory (auto-detected if not provided)")
@click.option("--save-dir", required=True, type=click.Path(), help="Output directory for visualizations")
@click.option("--classes", help="Comma-separated class names")
@click.option("--classes-file", type=click.Path(exists=True), help="File with class names (one per line)")
@click.option("--filter-classes", help="Only visualize these classes (comma-separated)")
@click.option("--no-labels", is_flag=True, default=False, help="Hide label text")
@click.option("--shuffle", is_flag=True, default=False, help="Randomize image order")
@click.option("--images", help="Only visualize specific images (comma-separated filenames)")
def visualize_command(
    source: str,
    input_format: str,
    img_dir: str | None,
    save_dir: str,
    classes: str | None,
    classes_file: str | None,
    filter_classes: str | None,
    no_labels: bool,
    shuffle: bool,
    images: str | None,
):
    """Visualize dataset annotations with bounding boxes.

    Examples:
        # Visualize all images
        lqcv visualize --source labels/ --format yolo --save-dir vis/ --classes "person,car"

        # Visualize specific classes only
        lqcv visualize --source xmls/ --format xml --save-dir vis/ --filter-classes "person,car"

        # Visualize specific images
        lqcv visualize --source data.json --format coco --save-dir vis/ --images "img1.jpg,img2.jpg"
    """
    # Parse class names
    class_names = None
    if classes:
        class_names = [c.strip() for c in classes.split(",")]
    elif classes_file:
        with open(classes_file) as f:
            class_names = [line.strip() for line in f if line.strip()]

    # Load converter
    click.echo(f"Loading {input_format.upper()} dataset from {source}...")
    converter = get_converter(input_format, source, class_names, img_dir)

    click.echo(f"Dataset loaded: {len(converter.labels)} images")

    # Parse filter classes
    filter_class_list = []
    if filter_classes:
        filter_class_list = [c.strip() for c in filter_classes.split(",")]

    # Parse image names
    image_names = None
    if images:
        image_names = [img.strip() for img in images.split(",")]

    # Visualize
    click.echo(f"Generating visualizations...")
    converter.visualize(
        save_dir=save_dir,
        classes=filter_class_list,
        show_labels=not no_labels,
        shuffle=shuffle,
        im_names=image_names,
    )

    click.echo(f"✅ Visualizations saved to {save_dir}")


@main.command("imshow")
@click.option("--source", required=True, help="Path to images directory")
@click.option("--shuffle", is_flag=True, default=False, help="Randomize image order")
@click.option("--nwindow", is_flag=True, default=False, help="Whether to use normalized window style")
def imshow_command(source: str, shuffle: bool, nwindow: bool):
    """Display dataset images with annotations in an interactive window."""
    from lqcv.tools.file import get_files
    from lqcv.utils.plot import cv2_imshow
    import cv2

    for file in get_files(source, shuffle=shuffle):
        im = cv2.imread(str(file))
        if im is None:
            continue
        cv2_imshow(im, nwindow=nwindow)


@main.command("info")
@click.option("--source", required=True, help="Path to labels directory (YOLO/XML) or JSON file (COCO)")
@click.option(
    "--format", "input_format", required=True, type=click.Choice(["yolo", "xml", "coco"]), help="Input format"
)
@click.option("--img-dir", type=click.Path(), help="Images directory (auto-detected if not provided)")
@click.option("--classes", help="Comma-separated class names")
@click.option("--classes-file", type=click.Path(exists=True), help="File with class names (one per line)")
def info_command(
    source: str,
    input_format: str,
    img_dir: str | None,
    classes: str | None,
    classes_file: str | None,
):
    """Generate dataset info with box size and category distribution.

    Examples:
        # Get info for YOLO dataset
        lqcv info --source labels/ --format yolo --classes "person,car,dog"

        # Get info using classes file
        lqcv info --source xmls/ --format xml --classes-file classes.txt

        # Get info for COCO dataset
        lqcv info --source annotations.json --format coco
    """
    # Parse class names
    class_names = None
    if classes:
        class_names = [c.strip() for c in classes.split(",")]
    elif classes_file:
        with open(classes_file) as f:
            class_names = [line.strip() for line in f if line.strip()]

    # Load converter
    click.echo(f"Loading {input_format.upper()} dataset from {source}...")
    converter = get_converter(input_format, source, class_names, img_dir)

    click.echo(f"Dataset loaded: {len(converter.labels)} images")
    click.echo(converter)

    # Generate info
    click.echo("\nGenerating dataset info...")
    converter.get_info()

    click.echo("✅ info saved to dataset_info.png")


if __name__ == "__main__":
    main()
