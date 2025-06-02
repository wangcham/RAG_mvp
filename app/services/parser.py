import PyPDF2
from docx import Document
import pandas as pd
import csv
import chardet
from typing import Union, List
import logging
import markdown
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
from html2text import html2text
import re

class FileParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)

    async def parse(self) -> str:
        """根据文件扩展名选择相应的解析方法"""
        file_extension = self.file_path.split('.')[-1].lower()
        parser_method = getattr(self, f'_parse_{file_extension}', None)
        
        if parser_method is None:
            await self.logger.error(f"Unsupported file format: {file_extension}")
            return None
        
        return await parser_method()

    async def _parse_txt(self) -> str:
        try:
            # 检测文件编码
            with open(self.file_path, 'rb') as file:
                raw_data = file.read()
                detected = chardet.detect(raw_data)
                encoding = detected['encoding'] or 'utf-8'
            
            # 使用检测到的编码读取文件
            with open(self.file_path, 'r', encoding=encoding) as file:
                print(file.read())
                return file.read()
        except Exception as e:
            self.logger.error(f"Error parsing TXT file: {str(e)}")
            raise
        
    async def _parse_pdf(self) -> str:
        try:
            text_content = []
            with open(self.file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text_content.append(page.extract_text())
            return '\n'.join(text_content)
        except Exception as e:
            self.logger.error(f"Error parsing PDF file: {str(e)}")
            raise
        
    async def _parse_docx(self) -> str:
        try:
            doc = Document(self.file_path)
            text_content = []
            for paragraph in doc.paragraphs:
                text_content.append(paragraph.text)
            return '\n'.join(text_content)
        except Exception as e:
            self.logger.error(f"Error parsing DOCX file: {str(e)}")
            raise
    
    async def _parse_doc(self) -> str:
        # 注意：.doc 文件需要使用其他库，如 antiword 或 textract
        # 这里建议将 .doc 转换为 .docx 后再处理
        self.logger.warning("Direct .doc parsing not supported. Please convert to .docx first.")
        raise NotImplementedError("Direct .doc parsing not supported. Please convert to .docx first.")
        
    async def _parse_xlsx(self) -> Union[str, List[str]]:
        try:
            # 读取所有工作表
            excel_file = pd.ExcelFile(self.file_path)
            text_content = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(self.file_path, sheet_name=sheet_name)
                # 将 DataFrame 转换为字符串
                sheet_text = f"Sheet: {sheet_name}\n{df.to_string()}"
                text_content.append(sheet_text)
            
            return '\n\n'.join(text_content)
        except Exception as e:
            self.logger.error(f"Error parsing XLSX file: {str(e)}")
            raise
        
    async def _parse_csv(self) -> str:
        try:
            # 检测文件编码
            with open(self.file_path, 'rb') as file:
                raw_data = file.read()
                detected = chardet.detect(raw_data)
                encoding = detected['encoding'] or 'utf-8'
            
            # 读取 CSV 文件
            df = pd.read_csv(self.file_path, encoding=encoding)
            return df.to_string()
        except Exception as e:
            self.logger.error(f"Error parsing CSV file: {str(e)}")
            raise
    
    async def _parse_markdown(self) -> str:
        try:
            # 检测文件编码
            with open(self.file_path, 'rb') as file:
                raw_data = file.read()
                detected = chardet.detect(raw_data)
                encoding = detected['encoding'] or 'utf-8'
            
            # 读取 Markdown 文件
            with open(self.file_path, 'r', encoding=encoding) as file:
                md_content = file.read()
            
            # 将 Markdown 转换为 HTML
            html_content = markdown.markdown(
                md_content,
                extensions=['extra', 'codehilite', 'tables', 'toc']
            )
            
            # 使用 BeautifulSoup 清理 HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 提取文本，保留基本结构
            text_content = []
            for element in soup.stripped_strings:
                # 保留标题的层级结构
                if element.parent.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    level = int(element.parent.name[1])
                    text_content.append('#' * level + ' ' + element)
                # 保留列表结构
                elif element.parent.name in ['ul', 'ol']:
                    text_content.append('* ' + element)
                # 保留代码块
                elif element.parent.name == 'pre':
                    text_content.append('```\n' + element + '\n```')
                else:
                    text_content.append(element)
            
            return '\n'.join(text_content)
        except Exception as e:
            self.logger.error(f"Error parsing Markdown file: {str(e)}")
            raise

    async def _parse_html(self) -> str:
        try:
            # 检测文件编码
            with open(self.file_path, 'rb') as file:
                raw_data = file.read()
                detected = chardet.detect(raw_data)
                encoding = detected['encoding'] or 'utf-8'
            
            # 读取 HTML 文件
            with open(self.file_path, 'r', encoding=encoding) as file:
                html_content = file.read()
            
            # 使用 BeautifulSoup 解析 HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 移除脚本和样式元素
            for script in soup(["script", "style"]):
                script.decompose()
            
            # 提取文本，保留基本结构
            text_content = []
            
            # 处理标题
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                level = int(heading.name[1])
                text_content.append('#' * level + ' ' + heading.get_text().strip())
            
            # 处理段落
            for p in soup.find_all('p'):
                text = p.get_text().strip()
                if text:
                    text_content.append(text)
            
            # 处理列表
            for ul in soup.find_all(['ul', 'ol']):
                for li in ul.find_all('li'):
                    text_content.append('* ' + li.get_text().strip())
            
            # 处理表格
            for table in soup.find_all('table'):
                # 提取表头
                headers = [th.get_text().strip() for th in table.find_all('th')]
                if headers:
                    text_content.append(' | '.join(headers))
                    text_content.append(' | '.join(['---'] * len(headers)))
                
                # 提取表格内容
                for row in table.find_all('tr'):
                    cells = [td.get_text().strip() for td in row.find_all('td')]
                    if cells:
                        text_content.append(' | '.join(cells))
            
            return '\n'.join(text_content)
        except Exception as e:
            self.logger.error(f"Error parsing HTML file: {str(e)}")
            raise
    
    async def _parse_epub(self) -> str:
        try:
            # 读取 EPUB 文件
            book = epub.read_epub(self.file_path)
            
            text_content = []
            
            # 提取元数据
            metadata = book.get_metadata('DC', 'title')
            if metadata:
                text_content.append(f"Title: {metadata[0][0]}")
            
            # 提取目录
            toc = book.get_toc()
            if toc:
                text_content.append("\nTable of Contents:")
                for item in toc:
                    if isinstance(item, tuple):
                        # 处理嵌套目录
                        chapter, subchapters = item
                        text_content.append(f"- {chapter.title}")
                        for subchapter in subchapters:
                            text_content.append(f"  - {subchapter.title}")
                    else:
                        text_content.append(f"- {item.title}")
            
            # 提取正文内容
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    # 将 HTML 内容转换为文本
                    html_content = item.get_content().decode('utf-8')
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # 移除脚本和样式元素
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # 提取文本
                    text = soup.get_text(separator='\n', strip=True)
                    # 清理多余的空白行
                    text = re.sub(r'\n\s*\n', '\n\n', text)
                    text_content.append(text)
            
            return '\n\n'.join(text_content)
        except Exception as e:
            self.logger.error(f"Error parsing EPUB file: {str(e)}")
            raise


