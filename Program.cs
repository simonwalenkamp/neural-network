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

    // Activation Function
    private static (Matrix<double> output, Matrix<double> cache) ComputeSigmoid(Matrix<double> input)
    {
        var activatedOutput = Matrix<double>.Build.Dense(input.RowCount, input.ColumnCount);
        var activationCache = Matrix<double>.Build.Dense(input.RowCount, input.ColumnCount);

        for (int row = 0; row < input.RowCount; row++)
        {
            for (int col = 0; col < input.ColumnCount; col++)
            {
                activatedOutput[row, col] = 1 / (1 + Math.Exp(-input[row, col]));
                activationCache[row, col] = input[row, col];
            }
        }

        return (activatedOutput, activationCache);
    }

    private static (Matrix<double>, List<((Matrix<double> previousInput, Matrix<double>, Matrix<double>) linearChache, Matrix<double> activationCache)>) ForwardPropagation(Matrix<double> input, Dictionary<string, Matrix<double>> parameters)
    {
        Matrix<double> layerInput = input;

        int numberOfLayers = parameters.Count / 2;
        List<((Matrix<double> previousInput, Matrix<double>, Matrix<double>) linearChache, Matrix<double> activationCache)> caches = [];

        for (int i = 1; i <= numberOfLayers; i++)
        {
            Matrix<double> previousInput = layerInput;
            Matrix<double> z = parameters[$"weights{i}"] * previousInput + parameters[$"biases{i}"];

            (Matrix<double> previousInput, Matrix<double>, Matrix<double>) linearCache = (previousInput, parameters[$"weights{i}"], parameters[$"biases{i}"]);

            (Matrix<double> output, Matrix<double> cache) result = ComputeSigmoid(z);

            layerInput = result.output;

            caches.Add((linearCache, result.cache));
        }

        return (layerInput, caches);
    }
}