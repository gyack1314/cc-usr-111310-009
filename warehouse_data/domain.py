"""核对原始配置及数量，不执行仓库变换。"""
from .io import unique
def summarize(v,root):
    unique(v['facts'],'fact_id');nodes=v['nodes']
    if len(nodes)!=len(set(nodes)):raise ValueError('节点重复')
    if any(e['child'] not in nodes or e['parent'] not in nodes for e in v['edges']):raise ValueError('边节点未知')
    return {'nodes':len(nodes),'edges':len(v['edges']),'facts':len(v['facts'])}
