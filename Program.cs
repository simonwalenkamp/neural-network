using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

internal class Program
{
    private static void Main()
    {
        Console.WriteLine("initializing parameters");

        int[] dimensions = [5, 4, 3];
        Dictionary<string, Matrix<double>> parameters = InitializeParameters(dimensions);

        foreach (KeyValuePair<string, Matrix<double>> paramenter in parameters)
        {
            Console.WriteLine($"{paramenter.Key} : {paramenter.Value}");
        }

    }

    private static Dictionary<string, Matrix<double>> InitializeParameters(int[] layerDimensions)
    {
        Random random = new(3);

        Dictionary<string, Matrix<double>> parameters = [];

        for (int i = 1; i < layerDimensions.Length; i++)
        {
            Matrix<double> weights = Matrix<double>.Build.Random(layerDimensions[i], layerDimensions[i - 1], new Normal(0, 0.01, random));
            Matrix<double> biases = Matrix<double>.Build.Random(layerDimensions[i], 1, 0);

            parameters["weights" + i] = weights;
            parameters["biases" + i] = biases;
        }

        return parameters;
    }
}