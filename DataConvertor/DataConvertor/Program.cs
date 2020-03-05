using System.Data.SQLite;
namespace DataConvertor
{
    
    internal class Program
    {
        public static void Main(string[] args)
        {
            SQLiteConnection.CreateFile("MyDataBase.dtbase");
            SQLiteCommand command = new SQLiteCommand();
            SQLiteDataReader dataReader = command.ExecuteReader();
            
            

        }
    }
}