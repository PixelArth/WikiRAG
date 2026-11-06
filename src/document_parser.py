"""
Turns documents into retrievable chunks of text.
"""

from pathlib import Path
import os
import json
import re

class MarkdownParser:
    """
    Turns markdown texts into chunks of texts
    """
    
    
    _EXTENSION = ".md"
     
    def __init__(self, dirPath : str, metaFilePath : str, cleanFilenames = False, exclude_categories : list[str] | None = []):
        
        self.mdDir = dirPath
        docsList = list(Path(self.mdDir).glob(pattern=f"*{self._EXTENSION}")) # Get only the desired format
        meta = self.openMetaFile(metaFilePath)
        if cleanFilenames:
            docsList = self.cleanFilenames(docsList, self.mdDir)
            
        self.docsList, self.meta = self.excludeCategories(docsList = docsList, meta=meta, catsToExclude=exclude_categories)

    def cleanFilenames(self, filenames : list[Path], dirPath : str):
        for filename in filenames:
            stem = filename.stem
            suffix = filename.suffix
            parent = filename.parent
            
            if "%21" in filename.stem:
                os.rename(src=filename, dst=parent / Path(stem.replace("%21","!") + suffix))
            if "%2B" in filename.stem:
                os.rename(src=filename, dst=parent / Path(stem.replace("%2B","+") + suffix))
            if "%27" in filename.stem:
                os.rename(src=filename, dst=parent / Path(stem.replace("%27","'") + suffix))
            if "%3A" in filename.stem:
                os.rename(src=filename, dst=parent / Path(stem.replace("%3A",":") + suffix))
        docsList = list(Path(dirPath).glob(pattern=f"*{self._EXTENSION}"))
        return docsList
        
    def openMetaFile(self, filepath : str):
        with open(filepath, encoding='utf-8', mode='r') as f:
            meta = json.load(f)
        return meta
    
    def excludeCategories(self,docsList : list[Path], meta : dict, catsToExclude : list[str]):
        """
        meta dict structure : { filename.stem : {"title" : ... , "category" : ...} }
        """
        page_names = [filename.stem for filename in docsList]
        n_pages = len(page_names)
        
        pageToExclude = []
        for pagename,metadata in meta.items():
            for cat in catsToExclude:
                if metadata["category"] is not None and cat in metadata["category"]:
                    pageToExclude.append(pagename)
                    break
        for pagename in pageToExclude:
            if pagename in page_names:
                docsList.remove(Path(self.mdDir) / (pagename+".md"))
                meta.pop(pagename)
        print(f"Removed {n_pages - len(docsList)} pages of unwanted categories : {catsToExclude}")
        return docsList,meta
        
    def split_sections(self, filename : Path, text : str) -> list[dict]:
        """
        Splits a markdown file into sections based on the '##' tags
        The sub-sections ('###' and below) stay in the text of their '##' section
        The text before the first '##' (if any) is the section '#main'
        The ids are made unique when a title appears several times in the file
        """

        sections = []
        titleCount = {}
        for chunk in re.split(r"(?m)^(?=## )", text):
            chunk = chunk.strip()
            if not chunk:
                continue

            if chunk.startswith("## "):
                title = chunk.splitlines()[0].removeprefix("## ").strip()
            else:
                title = "main"

            titleCount[title] = titleCount.get(title, 0) + 1
            suffix = "" if titleCount[title] == 1 else f"-{titleCount[title]}"

            sections.append({
                "id" : f"{filename.stem}#{title}{suffix}",
                "text" : f"{filename.stem.replace("_"," ")} - {title}\n" + chunk,
                "metadata" : self.meta[filename.stem],
            })

        return sections

    def chunkAllDocs(self) -> list[dict]:
        """
        Proceeds to the chunking of all documents listed in self.docsList
        """
        
        print("Chunking docs...")
        chunks = []
        
        for file in self.docsList:
            with open(file=file, mode='r', encoding='utf-8') as f:
                text = f.readlines()
                text = "".join(s for s in text)
                sections = self.split_sections(file, text)
                chunks += sections
        print("Done.")
        return chunks