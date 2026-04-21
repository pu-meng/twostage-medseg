# 先确认代理变量存在
echo $http_proxy
echo $HTTPS_PROXY

# 测试能不能通
curl -x http://127.0.0.1:7890 https://api.anthropic.com

# 先看7891的面板端口
curl http://127.0.0.1:9091/proxies  # 常见面板端口，试试
# 或者
curl http://127.0.0.1:9090/proxies


ps aux | grep -i clash

ss -tlnp | grep -E '7890|7891|7892'
# 或者
netstat -tlnp 2>/dev/null | grep clash



curl -X PUT http://127.0.0.1:9090/proxies/GLOBAL \
  -H "Content-Type: application/json" \
  -d '{"name":"🇺🇸 美国 ·原生 03"}'




# 1. 看有几个clash在跑，各自的配置文件
ps aux | grep clash

# 2. 每个clash的API端口和代理端口
sudo grep -E "external-controller|mixed-port" /path/to/each/config.yaml
ss -tlnp | grep clash


curl -s --noproxy '*' http://127.0.0.1:9090/proxies | python3 -m json.tool | grep -E '"name":|"now":' | head -40


# 获取所有节点名



# 3. 挨个端口测哪个能通
for port in 7890 7891 7892; do
  result=$(curl -s -x http://127.0.0.1:$port https://api.anthropic.com \
    -o /dev/null -w "%{http_code}" --connect-timeout 3)
  echo "port $port: $result"
done


curl -x http://127.0.0.1:7891 https://api.anthropic.com -v 2>&1 | head -20
# 常见位置
cat ~/.config/clash/config.yaml
# 或者
ls ~/.config/clash/
# 或者找进程确认配置路径
ps aux | grep clash

cat ~/qi77_clash.yaml | grep -A2 "mode\|port\|proxy-port"

grep -E "mode:|mixed-port:|port:|external-controller:" ~/qi77_clash.yaml | head -20

curl -s http://127.0.0.1:9091/proxies -H "Content-Type: application/json"

grep "secret" ~/qi77_clash.yaml

# 先杀掉当前clash进程
kill 742789

# 重启
nohup /usr/bin/clash -d /home/pumengyu/ -f /home/pumengyu/qi77_clash.yaml > /tmp/clash.log 2>&1 &


# 1. 杀掉旧进程
kill 2799310 && sleep 2 && nohup /usr/bin/clash -d /home/pumengyu/ -f /home/pumengyu/qi77_clash.yaml > /tmp/clash.log 2>&1 &
sleep 3 && cat /tmp/clash.log
[1] 2800717
nohup: ignoring input
18:13:47 INF [Config] initial compatible provider name=Proxy
18:13:47 INF inbound create success inbound=mixed addr=127.0.0.1:7891 network=tcp
18:13:47 INF inbound create success inbound=mixed addr=127.0.0.1:7891 network=udp
18:13:47 INF [API] listening addr=127.0.0.1:9091
(medicine) pumengyu@2080NoGUI:~/twostage_medseg$ 
# 2. 确认死了
ps aux | grep clash

# 3. 重启
nohup /usr/bin/clash -d /home/pumengyu/ -f /home/pumengyu/qi77_clash.yaml > /tmp/clash.log 2>&1 &

# 4. 看日志确认启动
sleep 3 && cat /tmp/clash.log


# 直接不走代理，测试节点服务器能不能通
curl -v --connect-timeout 5 mg1b.srooiar.top:38258
curl -v --connect-timeout 5 mg2.srooiar.top:58508

dpkg -i /home/pumengyu/2081/qiqi-4.6.1.deb