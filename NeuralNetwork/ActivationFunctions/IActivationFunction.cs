using System.Collections;

namespace NeuralNetwork.ActivationFunctions
{
    public interface IActivationFunction
    {
        double Activate(Node currentNode, Layer layer);
    }
}
