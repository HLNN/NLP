from py2neo import Graph

from config import Config


class Query:
    def __init__(self):
        self.graph = Graph(Config.url, user=Config.user, password=Config.password)

    def run(self, cql):
        result = []
        find_rela = self.graph.run(cql)
        for i in find_rela:
            result.append(i.items()[0][1])
        return result


if __name__ == '__main__':
    SQL = Query()
    cql = 'Match (m:Movie) where m.released > 2000 RETURN m limit 5'
    result = SQL.run(cql)
    print(result)
