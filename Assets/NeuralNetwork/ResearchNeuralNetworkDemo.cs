// http://channel9.msdn.com/Events/Build/2013/2-401
// http://research.microsoft.com/en-us/projects/neuralnetworks/BackPropDemo.aspx
// http://www.quaetrix.com/NeuralNetworkDemo.html

using System;

// C# console application demo of a neural network classification program using Visual Studio.
// Assumes you have Visual Studio (you can get a free 'Visual Studio Express' version).
// To build from command line, copy this code into notepad or similar editor and then save on 
// your local machine as file MSRNeuralProgram.cs. Next, launch the special Visual Studio
// command shell (it knows where the C# compiler is) and then cd-navigate to the directory
// where you saved this file. Then type 'csc.exe  MSRNeuralProgram.cs' and hit (Enter).
// The program will compile and create file MSRNeuralProgram.exe which you can run from the
// command line.
//
// The demo problem is to classify some dummy data. The 4 predictor values are just arbitrary
// numbers between 0 and 40. The value-to-predict is color, which can be either 'red',
// 'yellow', or 'blue'. The data is based on the famous Fisher's Iris data set.
//
// The neural network in this demo is 'normal' - it fully-connected and feed-forward.
// Training uses the back-propagation algorithm with momentum but no weight-decay, and 
// the behind-the-scenes error term is sum of squared errors (even though research suggests
// that cross-entropy is superior). The program does not perform input data normalization
// and so training is slower than it could be.

namespace ResearchNeuralNetworkDemo
{
  class MSRNeuralProgram
  {
    static void Main(string[] args)
    {
      Console.WriteLine("\nBegin neural network classification and prediction demo");
      Console.WriteLine("\nData is dummy, artificial data.");
      Console.WriteLine("X-data is x0, x1, x2, x3");
      Console.WriteLine("Y-data is Red = 0 0 1, Yellow = 0 1 0, Blue = 1 0 0 "); // 1-of-N encoded
      Console.WriteLine("The goal is to predict color from x0, x1, x2, x3\n");

      Console.WriteLine("Raw data resembles:\n");
      Console.WriteLine(" 25.6, 17.6,  7.1,  1.1,  Red");
      Console.WriteLine(" 35.1, 16.1, 23.6,  7.1,  Yellow");
      Console.WriteLine(" 31.6, 16.6, 30.1, 12.6,  Blue");
      Console.WriteLine(" ......\n");

      double[][] allData = new double[150][];
      allData[0] = new double[] { 25.6, 17.6, 7.1, 1.1, 0, 0, 1 };
      allData[1] = new double[] { 24.6, 15.1, 7.1, 1.1, 0, 0, 1 };
      allData[2] = new double[] { 23.6, 16.1, 6.6, 1.1, 0, 0, 1 };
      allData[3] = new double[] { 23.1, 15.6, 7.6, 1.1, 0, 0, 1 };
      allData[4] = new double[] { 25.1, 18.1, 7.1, 1.1, 0, 0, 1 };
      allData[5] = new double[] { 27.1, 19.6, 8.6, 2.1, 0, 0, 1 };
      allData[6] = new double[] { 23.1, 17.1, 7.1, 1.6, 0, 0, 1 };
      allData[7] = new double[] { 25.1, 17.1, 7.6, 1.1, 0, 0, 1 };
      allData[8] = new double[] { 22.1, 14.6, 7.1, 1.1, 0, 0, 1 };
      allData[9] = new double[] { 24.6, 15.6, 7.6, 0.6, 0, 0, 1 };
      allData[10] = new double[] { 27.1, 18.6, 7.6, 1.1, 0, 0, 1 };
      allData[11] = new double[] { 24.1, 17.1, 8.1, 1.1, 0, 0, 1 };
      allData[12] = new double[] { 24.1, 15.1, 7.1, 0.6, 0, 0, 1 };
      allData[13] = new double[] { 21.6, 15.1, 5.6, 0.6, 0, 0, 1 };
      allData[14] = new double[] { 29.1, 20.1, 6.1, 1.1, 0, 0, 1 };
      allData[15] = new double[] { 28.6, 22.1, 7.6, 2.1, 0, 0, 1 };
      allData[16] = new double[] { 27.1, 19.6, 6.6, 2.1, 0, 0, 1 };
      allData[17] = new double[] { 25.6, 17.6, 7.1, 1.6, 0, 0, 1 };
      allData[18] = new double[] { 28.6, 19.1, 8.6, 1.6, 0, 0, 1 };
      allData[19] = new double[] { 25.6, 19.1, 7.6, 1.6, 0, 0, 1 };
      allData[20] = new double[] { 27.1, 17.1, 8.6, 1.1, 0, 0, 1 };
      allData[21] = new double[] { 25.6, 18.6, 7.6, 2.1, 0, 0, 1 };
      allData[22] = new double[] { 23.1, 18.1, 5.1, 1.1, 0, 0, 1 };
      allData[23] = new double[] { 25.6, 16.6, 8.6, 2.6, 0, 0, 1 };
      allData[24] = new double[] { 24.1, 17.1, 9.6, 1.1, 0, 0, 1 };
      allData[25] = new double[] { 25.1, 15.1, 8.1, 1.1, 0, 0, 1 };
      allData[26] = new double[] { 25.1, 17.1, 8.1, 2.1, 0, 0, 1 };
      allData[27] = new double[] { 26.1, 17.6, 7.6, 1.1, 0, 0, 1 };
      allData[28] = new double[] { 26.1, 17.1, 7.1, 1.1, 0, 0, 1 };
      allData[29] = new double[] { 23.6, 16.1, 8.1, 1.1, 0, 0, 1 };
      allData[30] = new double[] { 24.1, 15.6, 8.1, 1.1, 0, 0, 1 };
      allData[31] = new double[] { 27.1, 17.1, 7.6, 2.1, 0, 0, 1 };
      allData[32] = new double[] { 26.1, 20.6, 7.6, 0.6, 0, 0, 1 };
      allData[33] = new double[] { 27.6, 21.1, 7.1, 1.1, 0, 0, 1 };
      allData[34] = new double[] { 24.6, 15.6, 7.6, 0.6, 0, 0, 1 };
      allData[35] = new double[] { 25.1, 16.1, 6.1, 1.1, 0, 0, 1 };
      allData[36] = new double[] { 27.6, 17.6, 6.6, 1.1, 0, 0, 1 };
      allData[37] = new double[] { 24.6, 15.6, 7.6, 0.6, 0, 0, 1 };
      allData[38] = new double[] { 22.1, 15.1, 6.6, 1.1, 0, 0, 1 };
      allData[39] = new double[] { 25.6, 17.1, 7.6, 1.1, 0, 0, 1 };
      allData[40] = new double[] { 25.1, 17.6, 6.6, 1.6, 0, 0, 1 };
      allData[41] = new double[] { 22.6, 11.6, 6.6, 1.6, 0, 0, 1 };
      allData[42] = new double[] { 22.1, 16.1, 6.6, 1.1, 0, 0, 1 };
      allData[43] = new double[] { 25.1, 17.6, 8.1, 3.1, 0, 0, 1 };
      allData[44] = new double[] { 25.6, 19.1, 9.6, 2.1, 0, 0, 1 };
      allData[45] = new double[] { 24.1, 15.1, 7.1, 1.6, 0, 0, 1 };
      allData[46] = new double[] { 25.6, 19.1, 8.1, 1.1, 0, 0, 1 };
      allData[47] = new double[] { 23.1, 16.1, 7.1, 1.1, 0, 0, 1 };
      allData[48] = new double[] { 26.6, 18.6, 7.6, 1.1, 0, 0, 1 };
      allData[49] = new double[] { 25.1, 16.6, 7.1, 1.1, 0, 0, 1 };
      allData[50] = new double[] { 35.1, 16.1, 23.6, 7.1, 0, 1, 0 };
      allData[51] = new double[] { 32.1, 16.1, 22.6, 7.6, 0, 1, 0 };
      allData[52] = new double[] { 34.6, 15.6, 24.6, 7.6, 0, 1, 0 };
      allData[53] = new double[] { 27.6, 11.6, 20.1, 6.6, 0, 1, 0 };
      allData[54] = new double[] { 32.6, 14.1, 23.1, 7.6, 0, 1, 0 };
      allData[55] = new double[] { 28.6, 14.1, 22.6, 6.6, 0, 1, 0 };
      allData[56] = new double[] { 31.6, 16.6, 23.6, 8.1, 0, 1, 0 };
      allData[57] = new double[] { 24.6, 12.1, 16.6, 5.1, 0, 1, 0 };
      allData[58] = new double[] { 33.1, 14.6, 23.1, 6.6, 0, 1, 0 };
      allData[59] = new double[] { 26.1, 13.6, 19.6, 7.1, 0, 1, 0 };
      allData[60] = new double[] { 25.1, 10.1, 17.6, 5.1, 0, 1, 0 };
      allData[61] = new double[] { 29.6, 15.1, 21.1, 7.6, 0, 1, 0 };
      allData[62] = new double[] { 30.1, 11.1, 20.1, 5.1, 0, 1, 0 };
      allData[63] = new double[] { 30.6, 14.6, 23.6, 7.1, 0, 1, 0 };
      allData[64] = new double[] { 28.1, 14.6, 18.1, 6.6, 0, 1, 0 };
      allData[65] = new double[] { 33.6, 15.6, 22.1, 7.1, 0, 1, 0 };
      allData[66] = new double[] { 28.1, 15.1, 22.6, 7.6, 0, 1, 0 };
      allData[67] = new double[] { 29.1, 13.6, 20.6, 5.1, 0, 1, 0 };
      allData[68] = new double[] { 31.1, 11.1, 22.6, 7.6, 0, 1, 0 };
      allData[69] = new double[] { 28.1, 12.6, 19.6, 5.6, 0, 1, 0 };
      allData[70] = new double[] { 29.6, 16.1, 24.1, 9.1, 0, 1, 0 };
      allData[71] = new double[] { 30.6, 14.1, 20.1, 6.6, 0, 1, 0 };
      allData[72] = new double[] { 31.6, 12.6, 24.6, 7.6, 0, 1, 0 };
      allData[73] = new double[] { 30.6, 14.1, 23.6, 6.1, 0, 1, 0 };
      allData[74] = new double[] { 32.1, 14.6, 21.6, 6.6, 0, 1, 0 };
      allData[75] = new double[] { 33.1, 15.1, 22.1, 7.1, 0, 1, 0 };
      allData[76] = new double[] { 34.1, 14.1, 24.1, 7.1, 0, 1, 0 };
      allData[77] = new double[] { 33.6, 15.1, 25.1, 8.6, 0, 1, 0 };
      allData[78] = new double[] { 30.1, 14.6, 22.6, 7.6, 0, 1, 0 };
      allData[79] = new double[] { 28.6, 13.1, 17.6, 5.1, 0, 1, 0 };
      allData[80] = new double[] { 27.6, 12.1, 19.1, 5.6, 0, 1, 0 };
      allData[81] = new double[] { 27.6, 12.1, 18.6, 5.1, 0, 1, 0 };
      allData[82] = new double[] { 29.1, 13.6, 19.6, 6.1, 0, 1, 0 };
      allData[83] = new double[] { 30.1, 13.6, 25.6, 8.1, 0, 1, 0 };
      allData[84] = new double[] { 27.1, 15.1, 22.6, 7.6, 0, 1, 0 };
      allData[85] = new double[] { 30.1, 17.1, 22.6, 8.1, 0, 1, 0 };
      allData[86] = new double[] { 33.6, 15.6, 23.6, 7.6, 0, 1, 0 };
      allData[87] = new double[] { 31.6, 11.6, 22.1, 6.6, 0, 1, 0 };
      allData[88] = new double[] { 28.1, 15.1, 20.6, 6.6, 0, 1, 0 };
      allData[89] = new double[] { 27.6, 12.6, 20.1, 6.6, 0, 1, 0 };
      allData[90] = new double[] { 27.6, 13.1, 22.1, 6.1, 0, 1, 0 };
      allData[91] = new double[] { 30.6, 15.1, 23.1, 7.1, 0, 1, 0 };
      allData[92] = new double[] { 29.1, 13.1, 20.1, 6.1, 0, 1, 0 };
      allData[93] = new double[] { 25.1, 11.6, 16.6, 5.1, 0, 1, 0 };
      allData[94] = new double[] { 28.1, 13.6, 21.1, 6.6, 0, 1, 0 };
      allData[95] = new double[] { 28.6, 15.1, 21.1, 6.1, 0, 1, 0 };
      allData[96] = new double[] { 28.6, 14.6, 21.1, 6.6, 0, 1, 0 };
      allData[97] = new double[] { 31.1, 14.6, 21.6, 6.6, 0, 1, 0 };
      allData[98] = new double[] { 25.6, 12.6, 15.1, 5.6, 0, 1, 0 };
      allData[99] = new double[] { 28.6, 14.1, 20.6, 6.6, 0, 1, 0 };
      allData[100] = new double[] { 31.6, 16.6, 30.1, 12.6, 1, 0, 0 };
      allData[101] = new double[] { 29.1, 13.6, 25.6, 9.6, 1, 0, 0 };
      allData[102] = new double[] { 35.6, 15.1, 29.6, 10.6, 1, 0, 0 };
      allData[103] = new double[] { 31.6, 14.6, 28.1, 9.1, 1, 0, 0 };
      allData[104] = new double[] { 32.6, 15.1, 29.1, 11.1, 1, 0, 0 };
      allData[105] = new double[] { 38.1, 15.1, 33.1, 10.6, 1, 0, 0 };
      allData[106] = new double[] { 24.6, 12.6, 22.6, 8.6, 1, 0, 0 };
      allData[107] = new double[] { 36.6, 14.6, 31.6, 9.1, 1, 0, 0 };
      allData[108] = new double[] { 33.6, 12.6, 29.1, 9.1, 1, 0, 0 };
      allData[109] = new double[] { 36.1, 18.1, 30.6, 12.6, 1, 0, 0 };
      allData[110] = new double[] { 32.6, 16.1, 25.6, 10.1, 1, 0, 0 };
      allData[111] = new double[] { 32.1, 13.6, 26.6, 9.6, 1, 0, 0 };
      allData[112] = new double[] { 34.1, 15.1, 27.6, 10.6, 1, 0, 0 };
      allData[113] = new double[] { 28.6, 12.6, 25.1, 10.1, 1, 0, 0 };
      allData[114] = new double[] { 29.1, 14.1, 25.6, 12.1, 1, 0, 0 };
      allData[115] = new double[] { 32.1, 16.1, 26.6, 11.6, 1, 0, 0 };
      allData[116] = new double[] { 32.6, 15.1, 27.6, 9.1, 1, 0, 0 };
      allData[117] = new double[] { 38.6, 19.1, 33.6, 11.1, 1, 0, 0 };
      allData[118] = new double[] { 38.6, 13.1, 34.6, 11.6, 1, 0, 0 };
      allData[119] = new double[] { 30.1, 11.1, 25.1, 7.6, 1, 0, 0 };
      allData[120] = new double[] { 34.6, 16.1, 28.6, 11.6, 1, 0, 0 };
      allData[121] = new double[] { 28.1, 14.1, 24.6, 10.1, 1, 0, 0 };
      allData[122] = new double[] { 38.6, 14.1, 33.6, 10.1, 1, 0, 0 };
      allData[123] = new double[] { 31.6, 13.6, 24.6, 9.1, 1, 0, 0 };
      allData[124] = new double[] { 33.6, 16.6, 28.6, 10.6, 1, 0, 0 };
      allData[125] = new double[] { 36.1, 16.1, 30.1, 9.1, 1, 0, 0 };
      allData[126] = new double[] { 31.1, 14.1, 24.1, 9.1, 1, 0, 0 };
      allData[127] = new double[] { 30.6, 15.1, 24.6, 9.1, 1, 0, 0 };
      allData[128] = new double[] { 32.1, 14.1, 28.1, 10.6, 1, 0, 0 };
      allData[129] = new double[] { 36.1, 15.1, 29.1, 8.1, 1, 0, 0 };
      allData[130] = new double[] { 37.1, 14.1, 30.6, 9.6, 1, 0, 0 };
      allData[131] = new double[] { 39.6, 19.1, 32.1, 10.1, 1, 0, 0 };
      allData[132] = new double[] { 32.1, 14.1, 28.1, 11.1, 1, 0, 0 };
      allData[133] = new double[] { 31.6, 14.1, 25.6, 7.6, 1, 0, 0 };
      allData[134] = new double[] { 30.6, 13.1, 28.1, 7.1, 1, 0, 0 };
      allData[135] = new double[] { 38.6, 15.1, 30.6, 11.6, 1, 0, 0 };
      allData[136] = new double[] { 31.6, 17.1, 28.1, 12.1, 1, 0, 0 };
      allData[137] = new double[] { 32.1, 15.6, 27.6, 9.1, 1, 0, 0 };
      allData[138] = new double[] { 30.1, 15.1, 24.1, 9.1, 1, 0, 0 };
      allData[139] = new double[] { 34.6, 15.6, 27.1, 10.6, 1, 0, 0 };
      allData[140] = new double[] { 33.6, 15.6, 28.1, 12.1, 1, 0, 0 };
      allData[141] = new double[] { 34.6, 15.6, 25.6, 11.6, 1, 0, 0 };
      allData[142] = new double[] { 29.1, 13.6, 25.6, 9.6, 1, 0, 0 };
      allData[143] = new double[] { 34.1, 16.1, 29.6, 11.6, 1, 0, 0 };
      allData[144] = new double[] { 33.6, 16.6, 28.6, 12.6, 1, 0, 0 };
      allData[145] = new double[] { 33.6, 15.1, 26.1, 11.6, 1, 0, 0 };
      allData[146] = new double[] { 31.6, 12.6, 25.1, 9.6, 1, 0, 0 };
      allData[147] = new double[] { 32.6, 15.1, 26.1, 10.1, 1, 0, 0 };
      allData[148] = new double[] { 31.1, 17.1, 27.1, 11.6, 1, 0, 0 };
      allData[149] = new double[] { 29.6, 15.1, 25.6, 9.1, 1, 0, 0 };
 
      Console.WriteLine("\nFirst 6 rows of entire 150-item data set:");
      ShowMatrix(allData, 6, 1, true);

      Console.WriteLine("Creating 80% training and 20% test data matrices");
      double[][] trainData = null;
      double[][] testData = null;
      MakeTrainTest(allData, out trainData, out testData);

      Console.WriteLine("\nFirst 5 rows of training data:");
      ShowMatrix(trainData, 5, 1, true);
      Console.WriteLine("First 3 rows of test data:");
      ShowMatrix(testData, 3, 1, true);

      // Data really should be normalized here!

      Console.WriteLine("\nCreating a 4-input, 7-hidden, 3-output neural network");
      Console.WriteLine("Hard-coded tanh for input-to-hidden and softmax for hidden-to-output activations");
      const int numInput = 4;
      const int numHidden = 7;
      const int numOutput = 3;
      NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);

      Console.WriteLine("\nInitializing weights and bias to small random values");
      nn.InitializeWeights();

      int maxEpochs = 4000;
      double learnRate = 0.01;
      double momentum = 0.001;
      
      Console.WriteLine("Setting maxEpochs = 4000, learnRate = 0.01, momentum = 0.001");
      Console.WriteLine("Training has hard-coded mean squared error < 0.020 stopping condition");

      Console.WriteLine("\nBeginning training using incremental back-propagation\n");
      nn.Train(trainData, maxEpochs, learnRate, momentum);
      Console.WriteLine("Training complete");

      double[] weights = nn.GetWeights();
      Console.WriteLine("Final neural network weights and bias values:");
      ShowVector(weights, 10, 3, true);

      double trainAcc = nn.Accuracy(trainData);
      Console.WriteLine("\nAccuracy on training data = " + trainAcc.ToString("F4"));

      double testAcc = nn.Accuracy(testData);
      Console.WriteLine("\nAccuracy on test data = " + testAcc.ToString("F4"));

      Console.WriteLine("\nEnd neural network demo\n");
      Console.ReadLine();

    } // Main

    static void MakeTrainTest(double[][] allData, out double[][] trainData, out double[][] testData)
    {
      // split allData into 80% trainData and 20% testData
      Random rnd = new Random(0);
      int totRows = allData.Length;
      int numCols = allData[0].Length;

      int trainRows = (int)(totRows * 0.80); // hard-coded 80-20 split
      int testRows = totRows - trainRows;

      trainData = new double[trainRows][];
      testData = new double[testRows][];

      int[] sequence = new int[totRows]; // create a random sequence of indexes
      for (int i = 0; i < sequence.Length; ++i)
        sequence[i] = i;

      for (int i = 0; i < sequence.Length; ++i)
      {
        int r = rnd.Next(i, sequence.Length);
        int tmp = sequence[r];
        sequence[r] = sequence[i];
        sequence[i] = tmp;
      }

      int si = 0; // index into sequence[]
      int j = 0; // index into trainData or testData

      for (; si < trainRows; ++si) // first rows to train data
      {
        trainData[j] = new double[numCols];
        int idx = sequence[si];
        Array.Copy(allData[idx], trainData[j], numCols);
        ++j;
      }

      j = 0; // reset to start of test data
      for (; si < totRows; ++si) // remainder to test data
      {
        testData[j] = new double[numCols];
        int idx = sequence[si];
        Array.Copy(allData[idx], testData[j], numCols);
        ++j;
      }
    } // MakeTrainTest

    static void Normalize(double[][] dataMatrix, int[] cols)
    {
      // in most cases you want to normalize the x-data
    }

    static void ShowVector(double[] vector, int valsPerRow, int decimals, bool newLine)
    {
      for (int i = 0; i < vector.Length; ++i)
      {
        if (i % valsPerRow == 0) Console.WriteLine("");
        Console.Write(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " ");
      }
      if (newLine == true) Console.WriteLine("");
    }

    static void ShowMatrix(double[][] matrix, int numRows, int decimals, bool newLine)
    {
      for (int i = 0; i < numRows; ++i)
      {
        Console.Write(i.ToString().PadLeft(3) + ": ");
        for (int j = 0; j < matrix[i].Length; ++j)
        {
          if (matrix[i][j] >= 0.0) Console.Write(" "); else Console.Write("-"); ;
          Console.Write(Math.Abs(matrix[i][j]).ToString("F" + decimals).PadRight(5) + " ");
        }
        Console.WriteLine("");
      }
      if (newLine == true) Console.WriteLine("");
    }

  } // class Program

  public class NeuralNetwork
  {
    private static Random rnd;

    private int numInput;
    private int numHidden;
    private int numOutput;

    private double[] inputs;

    private double[][] ihWeights; // input-hidden
    private double[] hBiases;
    private double[] hOutputs;

    private double[][] hoWeights; // hidden-output
    private double[] oBiases;

    private double[] outputs;

    // back-prop specific arrays (these could be local to method UpdateWeights)
    private double[] oGrads; // output gradients for back-propagation
    private double[] hGrads; // hidden gradients for back-propagation

    // back-prop momentum specific arrays (these could be local to method Train)
    private double[][] ihPrevWeightsDelta;  // for momentum with back-propagation
    private double[] hPrevBiasesDelta;
    private double[][] hoPrevWeightsDelta;
    private double[] oPrevBiasesDelta;


    public NeuralNetwork(int numInput, int numHidden, int numOutput)
    {
      rnd = new Random(0); // for InitializeWeights() and Shuffle()

      this.numInput = numInput;
      this.numHidden = numHidden;
      this.numOutput = numOutput;

      this.inputs = new double[numInput];

      this.ihWeights = MakeMatrix(numInput, numHidden);
      this.hBiases = new double[numHidden];
      this.hOutputs = new double[numHidden];

      this.hoWeights = MakeMatrix(numHidden, numOutput);
      this.oBiases = new double[numOutput];

      this.outputs = new double[numOutput];

      // back-prop related arrays below
      this.hGrads = new double[numHidden];
      this.oGrads = new double[numOutput];

      this.ihPrevWeightsDelta = MakeMatrix(numInput, numHidden);
      this.hPrevBiasesDelta = new double[numHidden];
      this.hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput);
      this.oPrevBiasesDelta = new double[numOutput];
    } // ctor

    private static double[][] MakeMatrix(int rows, int cols) // helper for ctor
    {
      double[][] result = new double[rows][];
      for (int r = 0; r < result.Length; ++r)
        result[r] = new double[cols];
      return result;
    }

    public override string ToString() // yikes
    {
      string s = "";
      s += "===============================\n";
      s += "numInput = " + numInput + " numHidden = " + numHidden + " numOutput = " + numOutput + "\n\n";

      s += "inputs: \n";
      for (int i = 0; i < inputs.Length; ++i)
        s += inputs[i].ToString("F2") + " ";
      s += "\n\n";

      s += "ihWeights: \n";
      for (int i = 0; i < ihWeights.Length; ++i)
      {
        for (int j = 0; j < ihWeights[i].Length; ++j)
        {
          s += ihWeights[i][j].ToString("F4") + " ";
        }
        s += "\n";
      }
      s += "\n";

      s += "hBiases: \n";
      for (int i = 0; i < hBiases.Length; ++i)
        s += hBiases[i].ToString("F4") + " ";
      s += "\n\n";

      s += "hOutputs: \n";
      for (int i = 0; i < hOutputs.Length; ++i)
        s += hOutputs[i].ToString("F4") + " ";
      s += "\n\n";

      s += "hoWeights: \n";
      for (int i = 0; i < hoWeights.Length; ++i)
      {
        for (int j = 0; j < hoWeights[i].Length; ++j)
        {
          s += hoWeights[i][j].ToString("F4") + " ";
        }
        s += "\n";
      }
      s += "\n";

      s += "oBiases: \n";
      for (int i = 0; i < oBiases.Length; ++i)
        s += oBiases[i].ToString("F4") + " ";
      s += "\n\n";

      s += "hGrads: \n";
      for (int i = 0; i < hGrads.Length; ++i)
        s += hGrads[i].ToString("F4") + " ";
      s += "\n\n";

      s += "oGrads: \n";
      for (int i = 0; i < oGrads.Length; ++i)
        s += oGrads[i].ToString("F4") + " ";
      s += "\n\n";

      s += "ihPrevWeightsDelta: \n";
      for (int i = 0; i < ihPrevWeightsDelta.Length; ++i)
      {
        for (int j = 0; j < ihPrevWeightsDelta[i].Length; ++j)
        {
          s += ihPrevWeightsDelta[i][j].ToString("F4") + " ";
        }
        s += "\n";
      }
      s += "\n";

      s += "hPrevBiasesDelta: \n";
      for (int i = 0; i < hPrevBiasesDelta.Length; ++i)
        s += hPrevBiasesDelta[i].ToString("F4") + " ";
      s += "\n\n";

      s += "hoPrevWeightsDelta: \n";
      for (int i = 0; i < hoPrevWeightsDelta.Length; ++i)
      {
        for (int j = 0; j < hoPrevWeightsDelta[i].Length; ++j)
        {
          s += hoPrevWeightsDelta[i][j].ToString("F4") + " ";
        }
        s += "\n";
      }
      s += "\n";

      s += "oPrevBiasesDelta: \n";
      for (int i = 0; i < oPrevBiasesDelta.Length; ++i)
        s += oPrevBiasesDelta[i].ToString("F4") + " ";
      s += "\n\n";

      s += "outputs: \n";
      for (int i = 0; i < outputs.Length; ++i)
        s += outputs[i].ToString("F2") + " ";
      s += "\n\n";

      s += "===============================\n";
      return s;
    }

    // ----------------------------------------------------------------------------------------

    public void SetWeights(double[] weights)
    {
      // copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
      int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
      if (weights.Length != numWeights)
        throw new Exception("Bad weights array length: ");

      int k = 0; // points into weights param

      for (int i = 0; i < numInput; ++i)
        for (int j = 0; j < numHidden; ++j)
          ihWeights[i][j] = weights[k++];
      for (int i = 0; i < numHidden; ++i)
        hBiases[i] = weights[k++];
      for (int i = 0; i < numHidden; ++i)
        for (int j = 0; j < numOutput; ++j)
          hoWeights[i][j] = weights[k++];
      for (int i = 0; i < numOutput; ++i)
        oBiases[i] = weights[k++];
    }

    public void InitializeWeights()
    {
      // initialize weights and biases to small random values
      int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
      double[] initialWeights = new double[numWeights];
      double lo = -0.01;
      double hi = 0.01;
      for (int i = 0; i < initialWeights.Length; ++i)
        initialWeights[i] = (hi - lo) * rnd.NextDouble() + lo;
      this.SetWeights(initialWeights);
    }

    public double[] GetWeights()
    {
      // returns the current set of wweights, presumably after training
      int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
      double[] result = new double[numWeights];
      int k = 0;
      for (int i = 0; i < ihWeights.Length; ++i)
        for (int j = 0; j < ihWeights[0].Length; ++j)
          result[k++] = ihWeights[i][j];
      for (int i = 0; i < hBiases.Length; ++i)
        result[k++] = hBiases[i];
      for (int i = 0; i < hoWeights.Length; ++i)
        for (int j = 0; j < hoWeights[0].Length; ++j)
          result[k++] = hoWeights[i][j];
      for (int i = 0; i < oBiases.Length; ++i)
        result[k++] = oBiases[i];
      return result;
    }

    // ----------------------------------------------------------------------------------------

    private double[] ComputeOutputs(double[] xValues)
    {
      if (xValues.Length != numInput)
        throw new Exception("Bad xValues array length");

      double[] hSums = new double[numHidden]; // hidden nodes sums scratch array
      double[] oSums = new double[numOutput]; // output nodes sums

      for (int i = 0; i < xValues.Length; ++i) // copy x-values to inputs
        this.inputs[i] = xValues[i];

      for (int j = 0; j < numHidden; ++j)  // compute i-h sum of weights * inputs
        for (int i = 0; i < numInput; ++i)
          hSums[j] += this.inputs[i] * this.ihWeights[i][j]; // note +=

      for (int i = 0; i < numHidden; ++i)  // add biases to input-to-hidden sums
        hSums[i] += this.hBiases[i];

      for (int i = 0; i < numHidden; ++i)   // apply activation
        this.hOutputs[i] = HyperTanFunction(hSums[i]); // hard-coded

      for (int j = 0; j < numOutput; ++j)   // compute h-o sum of weights * hOutputs
        for (int i = 0; i < numHidden; ++i)
          oSums[j] += hOutputs[i] * hoWeights[i][j];

      for (int i = 0; i < numOutput; ++i)  // add biases to input-to-hidden sums
        oSums[i] += oBiases[i];

      double[] softOut = Softmax(oSums); // softmax activation does all outputs at once for efficiency
      Array.Copy(softOut, outputs, softOut.Length);

      double[] retResult = new double[numOutput]; // could define a GetOutputs method instead
      Array.Copy(this.outputs, retResult, retResult.Length);
      return retResult;
    } // ComputeOutputs

    private static double HyperTanFunction(double x)
    {
      if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
      else if (x > 20.0) return 1.0;
      else return Math.Tanh(x);
    }

    private static double[] Softmax(double[] oSums) 
    {
      // does all output nodes at once so scale doesn't have to be re-computed each time
      // 1. determine max output sum
      double max = oSums[0];
      for (int i = 0; i < oSums.Length; ++i)
        if (oSums[i] > max) max = oSums[i];

      // 2. determine scaling factor -- sum of exp(each val - max)
      double scale = 0.0;
      for (int i = 0; i < oSums.Length; ++i)
        scale += Math.Exp(oSums[i] - max);

      double[] result = new double[oSums.Length];
      for (int i = 0; i < oSums.Length; ++i)
        result[i] = Math.Exp(oSums[i] - max) / scale;

      return result; // now scaled so that xi sum to 1.0
    }

    // ----------------------------------------------------------------------------------------

    private void UpdateWeights(double[] tValues, double learnRate, double momentum)
    {
      // update the weights and biases using back-propagation, with target values, eta (learning rate),
      // alpha (momentum)
      // assumes that SetWeights and ComputeOutputs have been called and so all the internal arrays and
      // matrices have values (other than 0.0)
      if (tValues.Length != numOutput)
        throw new Exception("target values not same Length as output in UpdateWeights");

      // 1. compute output gradients
      for (int i = 0; i < oGrads.Length; ++i)
      {
        // derivative of softmax = (1 - y) * y (same as log-sigmoid)
        double derivative = (1 - outputs[i]) * outputs[i]; 
        // 'mean squared error version'. research suggests cross-entropy is better here . . .
        oGrads[i] = derivative * (tValues[i] - outputs[i]); 
      }

      // 2. compute hidden gradients
      for (int i = 0; i < hGrads.Length; ++i)
      {
        double derivative = (1 - hOutputs[i]) * (1 + hOutputs[i]); // derivative of tanh = (1 - y) * (1 + y)
        double sum = 0.0;
        for (int j = 0; j < numOutput; ++j) // each hidden delta is the sum of numOutput terms
        {
          double x = oGrads[j] * hoWeights[i][j];
          sum += x;
        }
        hGrads[i] = derivative * sum;
      }

      // 3a. update hidden weights (gradients must be computed right-to-left but weights
      // can be updated in any order)
      for (int i = 0; i < ihWeights.Length; ++i) // 0..2 (3)
      {
        for (int j = 0; j < ihWeights[0].Length; ++j) // 0..3 (4)
        {
          double delta = learnRate * hGrads[j] * inputs[i]; // compute the new delta
          ihWeights[i][j] += delta; // update. note we use '+' instead of '-'. this can be very tricky.
          // add momentum using previous delta. on first pass old value will be 0.0 but that's OK.
          ihWeights[i][j] += momentum * ihPrevWeightsDelta[i][j]; 
          // weight decay would go here
          ihPrevWeightsDelta[i][j] = delta; // don't forget to save the delta for momentum 
        }
      }

      // 3b. update hidden biases
      for (int i = 0; i < hBiases.Length; ++i)
      {
        // the 1.0 below is the constant input for any bias; could leave out
        double delta = learnRate * hGrads[i] * 1.0; 
        hBiases[i] += delta;
        hBiases[i] += momentum * hPrevBiasesDelta[i]; // momentum
        // weight decay here
        hPrevBiasesDelta[i] = delta; // don't forget to save the delta
      }

      // 4. update hidden-output weights
      for (int i = 0; i < hoWeights.Length; ++i)
      {
        for (int j = 0; j < hoWeights[0].Length; ++j)
        {
          // see above: hOutputs are inputs to the nn outputs
          double delta = learnRate * oGrads[j] * hOutputs[i];  
          hoWeights[i][j] += delta;
          hoWeights[i][j] += momentum * hoPrevWeightsDelta[i][j]; // momentum
          // weight decay here
          hoPrevWeightsDelta[i][j] = delta; // save
        }
      }

      // 4b. update output biases
      for (int i = 0; i < oBiases.Length; ++i)
      {
        double delta = learnRate * oGrads[i] * 1.0;
        oBiases[i] += delta;
        oBiases[i] += momentum * oPrevBiasesDelta[i]; // momentum
        // weight decay here
        oPrevBiasesDelta[i] = delta; // save
      }
    } // UpdateWeights

    // ----------------------------------------------------------------------------------------

    public void Train(double[][] trainData, int maxEprochs, double learnRate, double momentum)
    {
      // train a back-prop style NN classifier using learning rate and momentum
      // no weight decay
      int epoch = 0;
      double[] xValues = new double[numInput]; // inputs
      double[] tValues = new double[numOutput]; // target values

      int[] sequence = new int[trainData.Length];
      for (int i = 0; i < sequence.Length; ++i)
        sequence[i] = i;

      while (epoch < maxEprochs)
      {
        double mse = MeanSquaredError(trainData);
        if (mse < 0.020) break; // consider passing value in as parameter
        //if (mse < 0.001) break; // consider passing value in as parameter

        Shuffle(sequence); // visit each training data in random order
        for (int i = 0; i < trainData.Length; ++i)
        {
          int idx = sequence[i];
          Array.Copy(trainData[idx], xValues, numInput); // extract x's and y's.
          Array.Copy(trainData[idx], numInput, tValues, 0, numOutput);
          ComputeOutputs(xValues); // copy xValues in, compute outputs (and store them internally)
          UpdateWeights(tValues, learnRate, momentum); // use back-prop to find better weights
        } // each training tuple
        ++epoch;
      }
    } // Train

    private static void Shuffle(int[] sequence)
    {
      for (int i = 0; i < sequence.Length; ++i)
      {
        int r = rnd.Next(i, sequence.Length);
        int tmp = sequence[r];
        sequence[r] = sequence[i];
        sequence[i] = tmp;
      }
    }

    private double MeanSquaredError(double[][] trainData) // used as a training stopping condition
    {
      // average squared error per training tuple
      double sumSquaredError = 0.0;
      double[] xValues = new double[numInput]; // first numInput values in trainData
      double[] tValues = new double[numOutput]; // last numOutput values

      for (int i = 0; i < trainData.Length; ++i) 
      {
        // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
        //  where the parens are not really there
        Array.Copy(trainData[i], xValues, numInput); // get xValues.
        Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // get target values
        double[] yValues = this.ComputeOutputs(xValues); // compute output using current weights
        for (int j = 0; j < numOutput; ++j)
        {
          double err = tValues[j] - yValues[j];
          sumSquaredError += err * err;
        }
      }

      return sumSquaredError / trainData.Length;
    }

    // ----------------------------------------------------------------------------------------

    public double Accuracy(double[][] testData)
    {
      // percentage correct using winner-takes all
      int numCorrect = 0;
      int numWrong = 0;
      double[] xValues = new double[numInput]; // inputs
      double[] tValues = new double[numOutput]; // targets
      double[] yValues; // computed Y

      for (int i = 0; i < testData.Length; ++i)
      {
        Array.Copy(testData[i], xValues, numInput); // parse test data into x-values and t-values
        Array.Copy(testData[i], numInput, tValues, 0, numOutput);
        yValues = this.ComputeOutputs(xValues);
        int maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?

        if (tValues[maxIndex] == 1.0) // ugly. consider AreEqual(double x, double y)
          ++numCorrect;
        else
          ++numWrong;
      }
      return (numCorrect * 1.0) / (numCorrect + numWrong); // ugly 2 - check for divide by zero
    }

    private static int MaxIndex(double[] vector) // helper for Accuracy()
    {
      // index of largest value
      int bigIndex = 0;
      double biggestVal = vector[0];
      for (int i = 0; i < vector.Length; ++i)
      {
        if (vector[i] > biggestVal)
        {
          biggestVal = vector[i]; bigIndex = i;
        }
      }
      return bigIndex;
    }

  } // NeuralNetwork

} // ns
