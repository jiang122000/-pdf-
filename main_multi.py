import os, json, time
from pathlib import Path
from openai import OpenAI 
from dotenv import load_dotenv 
load_dotenv()



client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)



PDF_DIR = Path("pdf_in")
OUT_DIR = Path("json_out")
OUT_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = (
    "你是一个文件解析助手. "
    "请用中文把整篇文章做结构化总结，输出纯 JSON，字段："
    "标题、正文、表格中的字段名和对应值（如“产品名称”、“产品代码”、“交易日”、“融资客户”等）。只返回 JSON。"
)

for pdf_path in PDF_DIR.glob("*.pdf"):
    print(f"\n正在处理 {pdf_path.name} ……")

    # 1) 上传
    file_obj = client.files.create(file=pdf_path, purpose="file-extract")
    file_id = file_obj.id
    print("   已上传，file_id =", file_id)

    # 2) 调用 qwen-long
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "system", "content": f"fileid://{file_id}"},
        {"role": "user", "content": SYSTEM_PROMPT}
    ]
    stream = client.chat.completions.create(
        model="qwen-long",
        messages=messages,
        stream=True,
        stream_options={"include_usage": True}
    )

    answer = ""
    for chunk in stream:
        # 防止 choices 为空导致 IndexError
        if chunk.choices and chunk.choices[0].delta.content:
            answer += chunk.choices[0].delta.content

    # 3) 保存
    json_name = pdf_path.stem + ".json"
    json_path = OUT_DIR / json_name
    try:
        parsed = json.loads(answer.strip())
    except json.JSONDecodeError:
        parsed = {"raw": answer.strip()}
    json_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   已保存 {json_path.name}")

print("\n全部完成！请查看 json_out 文件夹。")