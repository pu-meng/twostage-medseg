# 设置代理
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# 验证代理是否能出去
curl -I https://api.anthropic.com

# 然后启动 Claude Code
claude

curl -s https://api64.ipify.org

curl -s https://ipinfo.io

curl -I https://api.anthropic.com 2>&1 | grep cf-ray

find / -name "*.yaml" 2>/dev/null | grep clash

# 查看 clash 进程用的是哪个配置目录
ps aux | grep clash


# 先备份
sudo cp /srv/clash/config.yaml /srv/clash/config.yaml.bak

# 查看 Final 组和 GLOBAL 组的内容
sudo grep -A 10 "name: Final" /srv/clash/config.yaml
sudo grep -A 10 "name: GLOBAL" /srv/clash/config.yaml

sudo grep -A 5 "proxy-groups" /srv/clash/config.yaml | head -50

sudo grep "name:" /srv/clash/config.yaml

sudo grep "  - name:" /srv/clash/config.yaml | grep -i "美\|US\|美国\|洛\|纽"

sudo grep "^  - name:" /srv/clash/config.yaml | head -60


sudo sed -n '/name: AI/,/name: /p' /srv/clash/config.yaml | head -20

sudo grep "美国 01" /srv/clash/config.yaml | head -3
curl -X PUT http://127.0.0.1:9090/proxies/AI \
  -H "Content-Type: application/json" \
  -d '{"name": "🇺🇸 美国 01"}'


export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
ANTHROPIC_API_KEY=REDACTED_ANTHROPIC_API_KEY


curl -I https://api.anthropic.com 2>&1 | grep cf-ray

看看是什么地址


# 通过代理查出口 IP 和归属
curl -x http://127.0.0.1:7890 -s https://api.ipify.org
curl -x http://127.0.0.1:7890 -s https://ipinfo.io/json | python3 -m json.tool


# 加 -v 看 TLS 握手过程，加超时避免挂死
curl -x http://127.0.0.1:7890 -v --max-time 15 https://api.anthropic.com/v1/models \
  -H "anthropic-version: 2023-06-01" \
  -H "x-api-key: sk-ant-PLACEHOLDER"


# 查看 api.anthropic.com 走的是哪个规则/策略组
curl -s http://127.0.0.1:9090/connections | python3 -m json.tool | grep -A5 anthropic

# 或直接问 Clash API
curl -s "http://127.0.0.1:9090/proxies/AI" | python3 -m json.tool
# 看 "now" 字段是否真的是 美国01


curl -s --noproxy '*' http://127.0.0.1:9090/proxies | python3 -m json.tool | grep '"name"'

curl -s --noproxy '*' http://127.0.0.1:9090/proxies/AI | python3 -m json.tool


for i in 01 02 03 04 05 06 07 08 09 10 11 12 13; do
  node="🇺🇸 美国 $i"
  curl -s --noproxy '*' -X PUT "http://127.0.0.1:9090/proxies/AI" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"$node\"}" > /dev/null 2>&1
  sleep 0.5
  result=$(curl -x http://127.0.0.1:7890 -s --max-time 6 https://api.ipify.org 2>/dev/null)
  if [ -n "$result" ]; then
    echo "✅ $node → $result"
    break
  else
    echo "❌ $node"
  fi
done


curl -x http://127.0.0.1:7890 -s --max-time 10 https://api.anthropic.com/v1/models \
  -H "anthropic-version: 2023-06-01" \
  -H "x-api-key: test" 2>&1 | head -5


for region in "🇯🇵 日本" "🇸🇬 新加坡" "🇹🇼 台湾"; do
  for i in 01 02 03 04 05; do
    node="$region $i"
    curl -s --noproxy '*' -X PUT "http://127.0.0.1:9090/proxies/AI" \
      -H "Content-Type: application/json" \
      -d "{\"name\":\"$node\"}" > /dev/null 2>&1
    sleep 0.5
    result=$(curl -x http://127.0.0.1:7890 -s --max-time 6 https://api.ipify.org 2>/dev/null)
    if [ -n "$result" ]; then
      echo "✅ $node → $result"
    else
      echo "❌ $node"
    fi
  done
done



/usr/bin/clash -d /home/pumengyu/ -f /home/pumengyu/qi77_clash.yaml &
export HTTPS_PROXY=http://127.0.0.1:7891
export HTTP_PROXY=http://127.0.0.1:7891
export https_proxy=http://127.0.0.1:7891
export http_proxy=http://127.0.0.1:7891



# 下载新订阅替换配置
sudo curl -o /srv/clash/config.yaml 
sudo systemctl restart clash  # 或对应的重启命令

sleep 3
curl -x http://127.0.0.1:7890 -s --max-time 10 https://api.ipify.org && echo ""

sudo dpkg -i /home/pumengyu/2081/qiqi-4.6.1.deb