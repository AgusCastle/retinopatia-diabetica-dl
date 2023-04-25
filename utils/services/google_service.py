import gspread
from oauth2client.service_account import ServiceAccountCredentials

SCOPE = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/spreadsheets',
             'https://www.googleapis.com/auth/drive.file',
             'https://www.googleapis.com/auth/drive']

class GoogleService():

    def __init__(self):
        super()
        self.credentials = ServiceAccountCredentials.from_json_keyfile_name('utils/services/keys/python-results-384020-00965e2b0783.json', SCOPE)
        self.errors = []
        
    def insertRowToSheet(self, dictionary : dict):
        tmp = [dictionary['model'],
                dictionary['epoch'],
                dictionary['dataset'],
                dictionary['set'],
                dictionary['acc'],
                dictionary['aacc'],
                dictionary['aa02'],
                dictionary['aa134'],
                dictionary['none'],
                dictionary['mild'],
                dictionary['moderate'],
                dictionary['severe'],
                dictionary['proliferative']
                ]
        try:    
            gc = gspread.authorize(self.credentials)
            sheet = gc.open('Referencias_compartidas')
            worksheet = sheet.get_worksheet(1)
            worksheet.append_row(tmp)
        except:
            print('Ocurrio un error. Total de registros faltantes por subir: {}'.format(len(self.errors)))
            self.errors.append(tmp)

        finally:
            if len(self.errors) != 0:
                worksheet.append_rows(self.errors)
                self.errors = []
            print('Resultados actualizados en el Sheet.')
 
# Function Call
if __name__ == '__main__':

    gs = GoogleService()

    gs.insertRowToSheet({'model': 'convnext_1001', 'epoch': 3, 'dataset': 'DDR', 'set' : 'valid', 
                                    'acc': 89.12, 'aacc': 65.93, 'aa02': 65.93,'aa134': 65.93, 
                                    'none': 93.12, 'mild': 93.12, 'moderate': 93.12, 'severe': 93.12, 
                                    'proliferative': 93.12})