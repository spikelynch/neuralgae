import pystache, json, re

TEMPLATE = './templates/json.mustache'

def write_config(targets, outf):

    details = {}
    details['targets'] = ','.join([str(x) for x in targets])

    with open(TEMPLATE, 'r') as t:
        template = t.read()
        jscf = pystache.render(template, details)
        with open(outf, 'w') as o:
            o.write(jscf)
            print "Wrote to %s" % outf

def read_config(jsonfile):
    with open(jsonfile, 'r') as f:
        js = json.load(f)
        t = js['target']
        return [int(i) for i in t.split(',')]
    return None


class ImageNet(object):
    """A class for looking up ImageNet category names"""
    def __init__(self, classfile):
        self._names = {}
        re_class = re.compile('^(\d+) (n\d+) (.*)$')
        with open(classfile, 'r') as f:
            for l in f:
                m = re_class.search(l)
                if m:
                    i = int(m.group(1))
                    names = m.group(3)
                    self._names[i] = names.split(', ')
                else:
                    print "NO"

    def name(self, i):
        return self._names[i][0]

    def names(self, i):
        return self._names[i]
