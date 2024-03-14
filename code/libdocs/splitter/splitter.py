import glob
import logging
import os

from libdocs.metadata.metadata import Metadata
from PyPDF2 import PdfReader, PdfWriter


class Splitter:
    def __init__(
        self, input_dir: str, output_dir: str, files: list[str] = None
    ):
        dir_path = os.getcwd()
        if input_dir.startswith("/"):
            self.input_dir = input_dir
        else:
            self.input_dir = os.path.join(dir_path, input_dir)
        if not os.path.isdir(self.input_dir):
            raise Exception(
                f"input directory '{self.input_dir}' does not exist or is not a directory"
            )
        if output_dir.startswith("/"):
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join(dir_path, output_dir)
        if not os.path.isdir(self.output_dir):
            raise Exception(
                f"output directory '{self.output_dir}' does not exist or is not a directory"
            )
        if files is None or len(files) == 0:
            self.files = glob.glob(input_dir + "/*.pdf")
        else:
            self.files = []
            for file in files:
                f = os.path.join(self.input_dir, file)
                if not os.path.exists(f):
                    logging.warning(f"skipping non-existent input file '{f}")
                    continue
                if not f.endswith(".pdf"):
                    raise Exception(f"input file '{f}' is not a PDF file")
                self.files.append(f)

    def rename_and_load_meta(
        filepath, inputdir, outputdir: str
    ) -> (str, Metadata):

        filename = os.path.basename(filepath)

        # cleanup filename
        outfile = (
            filename.lower()
            .replace(" ", "_")
            .replace("+", "_")
            .replace("-", "_")
            .replace("__", "_")
            .replace("__", "_")
            .replace("._print", "")
        )

        outfilepath = filepath.replace(filename, outfile)

        # rename the file if needed
        if filepath != outfilepath:
            logging.debug(f"splitter: renaming {filename} -> {outfile}")
            os.rename(filepath, outfilepath)

        # get metafilepath, read
        metafilepath = outfilepath.replace(".pdf", ".metadata.json")
        logging.debug(f"splitter: eading metadata: {metafilepath}")
        meta = Metadata(metafilepath)

        # prepare output directory
        outfilepath = outfilepath.replace(".pdf", "").replace(
            inputdir, outputdir
        )
        os.makedirs(os.path.dirname(outfilepath), exist_ok=True)
        return outfilepath, meta

    def run(self) -> list[str]:
        ret = []
        numpages = 14

        for file in self.files:

            outfile, meta = Splitter.rename_and_load_meta(
                file, self.input_dir, self.output_dir
            )

            logging.info(f"splitter: partitioning {file}")
            reader = PdfReader(file)

            current = 0
            index = 0
            writers = [PdfWriter()]

            for page_index, page in enumerate(reader.pages):

                if not meta.allow_page(page_index):
                    continue

                if current < numpages:
                    current += 1
                else:
                    current = 0
                    index += 1
                    writers.append(PdfWriter())

                writers[index].add_page(page)

            for index, writer in enumerate(writers):
                partfile = f"{outfile}-{index}.pdf"
                logging.info(f"splitter: writing {partfile}")
                with open(partfile, "wb") as out:
                    writer.write(out)
                ret.append(partfile)

        return ret
