import os


# ------------- cleaning the dataset -------------
def create_block(data):
    clean_xml = []
    for line in data.splitlines():
        strriped = line.strip()
        if strriped:
            clean_xml.append(strriped)
    data = "".join(clean_xml)
    xml_block = (
        "<top>\n"
        + data.replace("<num>", "  <num>")
        .replace("<title>", "\n  </num>\n  <title>")
        .replace("<desc>", "\n  </title>\n  <desc>")
        .replace("<narr>", "\n  </desc>\n  <narr>")
        .replace("</top>", "\n  </narr>\n</top>")
        + "\n"
    )
    return xml_block


def read_dataset(file_path):
    with open(file_path, mode="r", encoding="utf-8") as file:
        dataset = file.read()
    dataset_blocks = dataset.split("<top>")
    xml_dataset = []
    for block in dataset_blocks:
        if "<num>" not in block:
            continue
        xml_block = create_block(block)
        xml_dataset.append(xml_block)
    xml_dataset = "<root>\n" + "".join(xml_dataset) + "\n </root>"
    dir_name, base_name = os.path.split(file_path)
    new_file_name = "xml_" + base_name + ".xml"
    xml_dataset_file_path = os.path.join(dir_name, new_file_name)
    with open(xml_dataset_file_path, mode="w", encoding="utf-8") as file:
        file.write(xml_dataset)


read_dataset(
    "/home/kk_gorbee/Documents/project/InternetRetrieval/BasicOperations/dataset/04.testset"
)
# -------------  -------------
