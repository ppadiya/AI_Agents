import pypandoc

# Automatically download Pandoc
pypandoc.download_pandoc()

def convert_md_to_docx(input_md_file, output_docx_file):
    try:
        pypandoc.convert_file(input_md_file, 'docx', outputfile=output_docx_file)
        print(f"Conversion successful! File saved at: {output_docx_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

input_file = "tailored_resume.md"  
output_file = "tailored_resume.docx"  
convert_md_to_docx(input_file, output_file)
