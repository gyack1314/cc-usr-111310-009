# 多父组织维度的带权闭包与重挂

此项目保存生产与供应链协同的湖仓原始资料。`warehouse_data.io`读取JSON清单及核对文件身份，`warehouse_data.domain`检查基本形状并输出数量，`warehouse_data.cli`展示摘要。[资料约定](docs/data-contract.md)定义字段、类型、时间、数量和版本范围。设备、批次、组织、事件和成本均为合成，不连接真实工厂或生产仓库。

Python3.11以上，无第三方Python依赖。项目根目录运行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q warehouse_data tests
python -m warehouse_data.cli inspect data/normal.json
```

`inspect`只读原始资料，校验多父层级与带路径证据的祖先桥，并输出按事实绑定版本的分摊；`rollup`判断所选节点能否作为集团总量；`version`预览局部改权后的新层级、重算闭包与重挂差额；`commit`把层级、桥、事实与重挂差额原子写入一个仓版本文档；`verify`从磁盘独立重算该文档。`invalid-config.json`有基本形状错误，`structural-cycle.json`会被结构环拒绝；其余清单有完整资料身份，但可能包含需要业务操作判断的冲突、缺口或不满足条件的边界。null与缺字段分别保留，有理数和decimal用字符串存储，不能先经float丢精度。

Dockerfile固定Linux镜像，`docker build -t warehouse-material .`构建；`docker run --rm warehouse-material`显示摘要；`docker run --rm warehouse-material python -m unittest discover -s tests -v`运行已有检查。所有资料本地可读，不需要账号、外部数据库或在线数据源。

```sh
python -m warehouse_data.cli bridge data/normal.json
python -m warehouse_data.cli rollup data/normal.json
python -m warehouse_data.cli version data/normal.json --new-version hv2
python -m warehouse_data.cli commit data/normal.json --out out/warehouse.json --warehouse-version wh1 --new-version hv2
python -m warehouse_data.cli verify out/warehouse.json
```

`bridge`列出每个祖先的精确权重、全部路径乘积证据与计算来源版本；`rollup`只有在
选中集合是全部根、节点间无祖先关系且每个事实流向选中根的权重和为1时才给出
`group_total`，否则列出不可直接加的原因。`commit`先完整重算核对，再以同目录临时
文件加 `os.replace` 单次替换；`--fail-after-staging` 仅用于验证失败时旧文档不变。
