# 4090服务器代理故障排查手册

## 服务器基本情况(4090,2026-04-13 深入了解)

### 三个 Clash 进程
| 进程 | 用户 | 配置文件 | HTTP端口 | 面板端口 | 协议 |
|------|------|----------|----------|----------|------|
| clash-1 | ZhaoPu | ~/qi77_clash.yaml | 7890 (127.0.0.1) | 未知 | HTTP |
| clash-2 | ZhaoPu | ~/1775529590563.yml | 7890冲突 | 未知 | HTTP |
| clash-3 | root | /srv/clash/config.yaml | 7891 (*所有网卡) | 9090 | socks5 |

### 关键结论
- **7890 = ZhaoPu 的 clash(HTTP代理)**,节点全是香港/新加坡,Anthropic封了
- **7891 = root 的 clash(socks5代理)**,有美国/日本节点,可以连Anthropic
- **9090 = root clash 的面板**,无secret,可直接访问
- **qi77 机场**：节点少(只有新加坡04/05-GPT、美国01/02-GPT),且目前全挂
- **root 机场(大哥云)**：节点多(香港/台湾/日本/美国/新加坡),美国原生/专线可用

### 最终解决方案
```
socks5://127.0.0.1:7891
    ↓ privoxy转换
http://127.0.0.1:8118
    ↓ Claude Code/curl使用
api.anthropic.com
```

---

## 下次遇到连不上,按顺序排查

### 第一步：快速诊断(30秒)
```bash
# 测试三个代理端口
for port in 7890 7891 8118; do
  r=$(curl -x http://127.0.0.1:$port https://api.anthropic.com \
    -o /dev/null -w "%{http_code}" --connect-timeout 5 2>/dev/null)
  echo "http $port: $r"
done
# socks5单独测
r=$(curl -x socks5://127.0.0.1:7891 https://api.anthropic.com \
  -o /dev/null -w "%{http_code}" --connect-timeout 5 2>/dev/null)
echo "socks5 7891: $r"
```

**结果解读：**
- `404` = 通了(Anthropic收到请求但路径不存在,正常)
- `000` = 没通(节点挂了或代理没起)
- `UnsupportedProxyProtocol` = 协议不对(Node.js不支持socks5)

### 第二步：确认代理进程是否在跑
```bash
ps aux | grep clash
ss -tlnp | grep -E '7890|7891|8118|9090'
```

### 第三步：切换节点(root clash面板)
```bash
# 看当前节点
curl --noproxy '*' -s http://127.0.0.1:9090/proxies | python3 -c "
import json,sys
d=json.load(sys.stdin)
for k,v in d['proxies'].items():
    if v.get('type')=='Selector':
        print(repr(k), '->', v.get('now'))
"

# 批量试所有可用节点
for node in '🇺🇸 美国 ·原生 03' '🇺🇸 美国 ·原生 04' '🇺🇸 美国-专线 · 01' '🇺🇸 美国-专线 · 02' '🇯🇵 日本 · 01' '🇯🇵 日本-专线 · 04'; do
  curl --noproxy '*' -X PUT http://127.0.0.1:9090/proxies/GLOBAL \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"$node\"}" > /dev/null 2>&1
  r=$(curl -x socks5://127.0.0.1:7891 https://api.anthropic.com \
    -o /dev/null -w "%{http_code}" --connect-timeout 8 2>/dev/null)
  echo "$node: $r"
done
```

### 第四步：privoxy 没起来
```bash
sudo systemctl status privoxy
sudo systemctl restart privoxy
# 确认配置里有这行
grep "forward-socks5" /etc/privoxy/config
# 没有就加
echo "forward-socks5 / 127.0.0.1:7891 ." | sudo tee -a /etc/privoxy/config
sudo systemctl restart privoxy
```

### 第五步：环境变量没设置
```bash
echo $HTTPS_PROXY  # 应该是 http://127.0.0.1:8118
# 没有就手动设
export http_proxy=http://127.0.0.1:8118
export https_proxy=http://127.0.0.1:8118
export HTTP_PROXY=http://127.0.0.1:8118
export HTTPS_PROXY=http://127.0.0.1:8118
```

---

## 常见错误含义速查

| 错误 | 含义 | 解决 |
|------|------|------|
| `000` | 连接在到达服务器前断了 | 节点挂了,切节点 |
| `404` | 服务器收到请求,路径不存在 | **正常！代理通了** |
| `Connection reset by peer` | 上游节点拒绝连接 | 切美国节点 |
| `502 Bad Gateway` | curl走了代理去访问面板 | 加 `--noproxy '*'` |
| `ECONNRESET` | Node.js连接被重置 | 检查节点/privoxy |
| `UnsupportedProxyProtocol` | Claude Code不支持socks5 | 用privoxy转http |
| `Exit 255` | clash启动失败 | `cat ~/clash.log`看原因 |

---

## 名词解释

- **inbound(入站)**：clash在本地开的"入口",接收你程序发来的流量
- **outbound(出站)**：clash把流量转发到远端节点
- **Selector**：手动选择节点的组,可以通过面板API切换
- **HTTP代理**：明文转发,Claude Code/Node.js支持
- **socks5代理**：更底层的代理协议,Node.js不支持
- **privoxy**：把socks5转成HTTP的中间件
- **INF**：Info日志,正常信息
- **Keep-Alive**：TCP连接复用,和故障无关

---

## 复制给Claude快速定位问题的模板

```
代理故障,服务器：4090 (PuMengYu)
诊断结果：
- http 7890: [结果]
- http 8118: [结果]  
- socks5 7891: [结果]
clash进程：[ps aux | grep clash 输出]
当前HTTPS_PROXY：[echo $HTTPS_PROXY]
报错信息：[具体错误]
```