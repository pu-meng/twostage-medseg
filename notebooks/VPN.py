
import json, subprocess, time

# 获取所有节点
result = subprocess.run(
    ['curl', '-s', '--noproxy', '*', 'http://127.0.0.1:9090/proxies'],
    capture_output=True, text=True
)
data = json.loads(result.stdout)

# 只要真实节点，不要组
skip_types = {'Selector', 'URLTest', 'Fallback', 'LoadBalance'}
nodes = [k for k, v in data['proxies'].items() if v.get('type') not in skip_types]

print(f"共 {len(nodes)} 个节点，开始测试...\n")

for node in nodes:
    # 切换节点
    payload = json.dumps({"name": node})
    subprocess.run(
        ['curl', '-s', '--noproxy', '*', '-X', 'PUT',
         'http://127.0.0.1:9090/proxies/GLOBAL',
         '-H', 'Content-Type: application/json', '-d', payload],
        capture_output=True
    )
    time.sleep(0.5)
    
    # 测试
    r = subprocess.run(
        ['curl', '-s', '-x', 'http://127.0.0.1:7891',
         'https://api.anthropic.com', '-o', '/dev/null',
         '-w', '%{http_code}', '--connect-timeout', '5'],
        capture_output=True, text=True
    )
    code = r.stdout.strip()
    
    if code != '000':
        print(f"✅ 可用: {node} (HTTP {code})")
    else:
        print(f"❌ 不通: {node}")

