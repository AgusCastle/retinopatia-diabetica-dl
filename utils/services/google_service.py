import gspread
from oauth2client.service_account import ServiceAccountCredentials

SCOPE = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/spreadsheets',
             'https://www.googleapis.com/auth/drive.file',
             'https://www.googleapis.com/auth/drive']

class GoogleService():

    def __init__(self):
        super()
        self.credentials = ServiceAccountCredentials.from_json_keyfile_name('./utils/services/keys/python-results-384020-00965e2b0783.json', SCOPE)
        self.errors = []
        
    def insertRowToSheet(self, array : list):
        try:    
            gc = gspread.authorize(self.credentials)
            sheet = gc.open('Referencias_compartidas')
            worksheet = sheet.get_worksheet(1)
            worksheet.append_row(array)
        except:
            print('Ocurrio un error. Total de registros faltantes por subir: {}'.format(len(self.errors)))
            self.errors.append(array)

        finally:
            if len(self.errors) != 0:
                gc = gspread.authorize(self.credentials)
                sheet = gc.open('Referencias_compartidas')
                worksheet = sheet.get_worksheet(1)
                worksheet.append_rows(self.errors)
                self.errors = []
            print('Resultados actualizados en el Sheet.')
 
# Function Call
if __name__ == '__main__':

    gs = GoogleService()

    gs.insertRowToSheet([12,12])