"""Save reproducible public-index queries for state/command/phase prior art.

Firecrawl was checked first and was unauthenticated. The fallback uses public
arXiv and OpenAlex APIs; its receipts never equate inaccessible sources or empty
results with proof of novelty. No API keys or global proxy changes are made.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from scripts import g1_semantic_split_study as study

QUERIES = [
    ('arxiv_state_command', 'arxiv', 'all:locomotion AND (all:"command encoder" OR all:"state encoder" OR all:"task-conditioned")'),
    ('arxiv_humanoid_fusion', 'arxiv', 'all:humanoid AND (all:"late fusion" OR all:conditioning OR all:"dual stream")'),
    ('arxiv_compact', 'arxiv', 'all:locomotion AND (all:"parameter efficient" OR all:"compact policy" OR all:"network architecture")'),
    ('openalex_state_command', 'openalex', 'locomotion separate state command encoder'),
    ('openalex_humanoid_phase', 'openalex', 'humanoid proprioception command phase policy network'),
]


def query_one(job, output):
    name, provider, query = job
    params = ({'search_query':query,'start':0,'max_results':12,'sortBy':'relevance'} if provider=='arxiv'
              else {'search':query,'per-page':12})
    base = 'https://export.arxiv.org/api/query?' if provider=='arxiv' else 'https://api.openalex.org/works?'
    url = base+urllib.parse.urlencode(params)
    receipt = {'query':query,'provider':provider,'url':url,'started_at':study.now(),
               'proxy_mode':'per-request direct connection; system proxy left unchanged'}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        request = urllib.request.Request(url,headers={'User-Agent':'G1SemanticSplitResearch/1.0'})
        with opener.open(request,timeout=35) as response:
            body=response.read();receipt['http_status']=response.status
        raw=output/(name+('.xml' if provider=='arxiv' else '.json'))
        raw.write_bytes(body)
        receipt.update(status='RETRIEVED',retrieved_at=study.now(),file=raw.name,sha256=study.sha(raw))
        if provider=='arxiv':
            namespace={'a':'http://www.w3.org/2005/Atom'}
            papers=[{'title':' '.join(e.findtext('a:title',default='',namespaces=namespace).split()),
                     'url':e.findtext('a:id',namespaces=namespace),
                     'published':e.findtext('a:published',namespaces=namespace),
                     'abstract':' '.join(e.findtext('a:summary',default='',namespaces=namespace).split())}
                    for e in ET.fromstring(body).findall('a:entry',namespace)]
        else:
            papers=[]
            for row in json.loads(body)['results']:
                inverted=row.get('abstract_inverted_index') or {}
                positions={p:word for word,indices in inverted.items() for p in indices}
                papers.append({'title':row['title'],'url':row.get('doi') or row['id'],
                               'openalex_id':row['id'],'year':row['publication_year'],
                               'abstract':' '.join(positions[p] for p in sorted(positions)),
                               'primary_url':(row.get('primary_location') or {}).get('landing_page_url')})
        receipt['records']=papers
    except Exception as error:
        receipt.update(status='UNAVAILABLE',ended_at=study.now(),error=repr(error))
    study.write(output/(name+'.receipt.json'),receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=study.DEFAULT_OUT/'literature')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        records=list(pool.map(lambda job:query_one(job,args.output),QUERIES))
    study.write(args.output/'search_manifest.json',{'at':study.now(),'queries':records,
        'firecrawl_status':'Not authenticated on 2026-10-26; public APIs used as fallback',
        'verdict':'UNKNOWN pending paper-level architecture verification; search results do not prove absence of prior art'})
    print(json.dumps([{'provider':r['provider'],'status':r['status'],'records':len(r.get('records',[]))} for r in records]))


if __name__=='__main__':main()
