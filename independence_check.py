import requests
import pandas as pd
import re

apikey = "4abc7a9f-70d1-4891-b741-6dbd99cb5c78"
version = "current"


def retrieve_neighbours(source, identifier, operation):
    df = pd.DataFrame(columns=['ui', 'uri', 'name', 'source'])
    uri = "https://uts-ws.nlm.nih.gov"
    content_endpoint = "/rest/content/" + version + "/source/" + source + "/" + identifier + "/" + operation

    pageNumber = 0
    try:
        while True:
            pageNumber += 1
            query = {'apiKey': apikey, 'pageNumber': pageNumber}
            r = requests.get(uri + content_endpoint, params=query)
            r.encoding = 'utf-8'
            items = r.json()

            if r.status_code != 200:
                break

            for result in items["result"]:
                entry = {}
                try:
                    entry['ui'] = result["ui"]
                except:
                    NameError
                try:
                    entry['uri'] = result["uri"]
                except:
                    NameError
                try:
                    entry['name'] = result["name"]
                except:
                    NameError
                try:
                    entry['source'] = result["source"]
                except:
                    NameError
                df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)

    except Exception as except_error:
        print(except_error)
    return df


def results_list(query):
    df = pd.DataFrame(columns=['ui', 'uri', 'name', 'source'])
    uri = "https://uts-ws.nlm.nih.gov"
    content_endpoint = "/rest/search/" + version
    full_url = uri + content_endpoint
    page = 0

    try:
        while True:
            page += 1
            query = {'string': query, 'apiKey': apikey, 'pageNumber': page}
            query['searchType'] = "normalizedString"
            r = requests.get(full_url, params=query)
            r.raise_for_status()
            print(r.url)
            r.encoding = 'utf-8'
            outputs = r.json()

            items = (([outputs['result']])[0])['results']

            if len(items) == 0:
                if page == 1:
                    break
                else:
                    break

            for result in items:
                entry = {
                    'ui': result['ui'],
                    'uri': result['uri'],
                    'name': result['name'],
                    'source': result['rootSource']
                }
                df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)


    except Exception as except_error:
        print(except_error)
    return df


def retrieve_atoms(identifier):
    uri = 'https://uts-ws.nlm.nih.gov'
    page = 0
    content_endpoint = '/rest/content/' + str(version) + '/CUI/' + str(identifier)
    query = {'apiKey': apikey, 'language': 'ENG'}
    r = requests.get(uri + content_endpoint, params=query)
    r.encoding = 'utf-8'

    if r.status_code != 200:
        raise Exception('Search term ' + "'" + str(identifier) + "'" + ' not found')

    items = r.json()
    jsonData = items['result']
    Atoms = jsonData['atoms']

    atoms_df = pd.DataFrame(columns=['Name', 'CUI', 'AUI', 'Term Type', 'Code', 'Source Vocabulary'])
    # The below uses the atoms' URI value from above as the starting URI.
    try:
        while True:
            page += 1
            atom_query = {'apiKey': apikey, 'pageNumber': page}
            a = requests.get(Atoms, params=atom_query)
            a.encoding = 'utf-8'

            if a.status_code != 200:
                break

            all_atoms = a.json()
            jsonAtoms = all_atoms['result']

            for atom in jsonAtoms:
                atom = {
                    "Name": atom['name'],
                    "CUI": jsonData['ui'],
                    "AUI": atom['ui'],
                    "Term Type": atom['termType'],
                    "Code": atom['code'],
                    "Source Vocabulary": atom['rootSource']
                }
                atoms_df = pd.concat([atoms_df, pd.DataFrame([atom])], ignore_index=True)
    except Exception as except_error:
        print(except_error)
    return atoms_df


if __name__ == "__main__":
    query = "prostate-cancer"
    cui_df = results_list(query)

    atoms_df = pd.DataFrame(columns=['Name', 'CUI', 'AUI', 'Term Type', 'Code', 'Source Vocabulary'])
    for index, row in cui_df.iterrows():
        atoms_df = pd.concat([atoms_df, retrieve_atoms(row['ui'])], ignore_index=True)

    parents_df = pd.DataFrame(columns=['ui', 'uri', 'name', 'source'])
    for index, row in atoms_df.iterrows():
        pattern = r"/source/(\w+)/(\w+)$"
        match = re.search(pattern, row['Code'])
        if match:
            source = match.group(1)
            identifier = match.group(2)

            operation = "parents"
            parents_df = pd.concat([parents_df, retrieve_neighbours(source, identifier, operation)], ignore_index=True)

    parent_list = parents_df['name'].to_list()
    print(parent_list)
    for word in ['male', 'female', 'black', 'white', 'asian', 'hispanic']:
        word_count = sum(len(re.findall(rf'\b{re.escape(word)}\b', sentence, re.IGNORECASE)) for sentence in parent_list)
        print(f'{word}: {word_count}')