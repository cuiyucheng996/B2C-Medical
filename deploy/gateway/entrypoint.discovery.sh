#!/bin/sh
set -e
CTMPL_CONSUL="${CONSUL_HTTP_ADDR:-host.docker.internal:8500}"
TEMPLATE="/etc/consul-template/upstreams.ctmpl"
OUT="/etc/nginx/conf.d/00-upstreams.conf"

consul-template -consul-addr="$CTMPL_CONSUL" -template "$TEMPLATE:$OUT" -once 2>/dev/null || \
  consul-template -consul-addr="$CTMPL_CONSUL" -template "$TEMPLATE:$OUT" -once -consul-retry-attempts=3

consul-template -consul-addr="$CTMPL_CONSUL" -template "$TEMPLATE:$OUT:nginx -s reload" &

exec nginx -g 'daemon off;'
