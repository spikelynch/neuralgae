import pystache, json, re, os.path

TEMPLATE = './Templates/json.mustache'

CLASS_RE = {
    'googlenet': '^(\d+) n\d+ (.*)$',
    'caffenet': '^(\d+) n\d+ (.*)$',
    'places': '^(\d+) (\S*)$',
    'manga_tag': '^(\d+) (.*)$'
    }

def write_config(details, outf):
    with open(outf, 'w') as o:
        json.dump(details, o)
        print "Wrote to %s" % outf

def read_config(jsonfile):
    with open(jsonfile, 'r') as f:
        js = json.load(f)
        return js['target']
    return None


class ImageCategories(object):
    """A class for looking up category names"""
    def __init__(self, model):
        self.model = model
        self.names = {}
        classfile = os.path.join('./Classes', '%s.txt' % model)
        if os.path.isfile(classfile) and model in CLASS_RE:
            re_class = re.compile(CLASS_RE[model])
            with open(classfile, 'r') as f:
                for l in f:
                    m = re_class.search(l)
                    if m:
                        i = int(m.group(1))
                        names = m.group(2)
                        self.names[i] = names.split(', ')
                    else:
                        print("Mismatched line {}".format(l))
        else:
            self.names = None

    def name(self, i):
        if self.names:
            return self.names[i][0]
        else:
            return str(i)

    def names(self, i):
        if self.names:
            return self.names[i]
        else:
            return [ str(i) ]
