#### 1、准备一个放 python代码的文件夹

如 D:\n8ndata\n8n-scripts

#### 2、安装 3方库  flask

```
pip  install  flask
```



#### 3、在该目录下创建 `api_server.py` 文件

写入 ，注意更改 `所有脚本存放根目录（Windows路径）` 和 `Python解释器路径`

```python
from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

# ====================== 配置区 自行修改 ======================
# 所有脚本存放根目录（Windows路径）
SCRIPT_ROOT = r"D:\n8ndata\n8n-scripts"
# Python解释器路径，执行where python找到
PYTHON_EXE = r"D:\Python311\python.exe"
# ==========================================================

@app.route("/run", methods=["POST"])
def run_script():
    req = request.get_json()
    if not req:
        return jsonify({"error": "需要JSON参数"}), 400

    # 必填：脚本文件名，例如 test.py
    script_name = req.get("script_name")
    # 可选：传入脚本参数，数组形式 ["arg1","arg2"]
    args = req.get("args", [])

    if not script_name:
        return jsonify({"error": "缺少 script_name 参数"}), 400

    # 拼接完整脚本路径（防止路径穿越攻击）
    script_path = os.path.abspath(os.path.join(SCRIPT_ROOT, script_name))
    if not script_path.startswith(os.path.abspath(SCRIPT_ROOT)):
        return jsonify({"error": "禁止访问脚本目录外文件"}), 403
    if not os.path.exists(script_path):
        return jsonify({"error": f"脚本不存在: {script_path}"}), 404

    # 构造执行命令
    cmd = [PYTHON_EXE, script_path] + args
    try:
        result = subprocess.run(
                    cmd,
                    capture_output=True,
                    encoding="gbk",
                    timeout=300,
                    errors="replace"
                )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "脚本执行超时"}), 504

    return jsonify({
        "success": True,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    })

if __name__ == "__main__":
    # 必须0.0.0.0，否则docker容器无法连接
    app.run(host="0.0.0.0", port=8123, debug=False)

```

#### 4、在当前目录开启 cmd , 用 python 执行 api_server.py 文件开启web服务

注意 cmd 不要关闭，观看自己的 ip地址，就是我们在 n8n中发起请求相关的地址

![image-20260816110552979](n8n通过服务的方式运行python文件.assets/image-20260816110552979.png)

在 n8n 中 发起 http请求的地址(以自己的为准) 或者用 固定的 host.docker.internal,注意结尾 加  /run

```
http://192.168.2.2:8123/run
或
http://host.docker.internal:8123/run
```

请求body,内指定要指定的 .py文件，该文件第一个 print 的结果，会被获取到

![image-20260816110923324](n8n通过服务的方式运行python文件.assets/image-20260816110923324.png)



#### 注意：如果返回值有乱码，就是 服务端中的代码编码设置问题

![image-20260816111942740](n8n通过服务的方式运行python文件.assets/image-20260816111942740.png)

用 gbk 有乱码的，改为 utf-8  。反之亦然。

更改后保存。记得关闭服务，重开这个服务。