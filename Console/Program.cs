using MlNet;

public static class Program
{
    static void Main()
    {
        //PerceptronTest.Start();
        //ImageTest.Start();
        MlNetTest.Start();
        MlNetTest.TestForOne();
        Console.ReadKey();
    }


    //static void WriteTable<T>(IEnumerable<T> table)
    //{
    //    ConsoleTable consoleTable = ConsoleTable.From(table);
    //    consoleTable.Options.EnableCount = false;
    //    consoleTable.Write();
    //}
}
