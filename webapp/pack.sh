#! /bin/bash
tar -I 'zstd' -h -cvf nlp_pack_$(date +"%y-%m-%d-%H-%M-%S").tar.zst backend nginx docker-compose.yaml