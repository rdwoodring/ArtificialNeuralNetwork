using NeuralNetwork.ActivationFunctions;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public class Node
    {
        public double Input { get; set; }
        public double? ActivatedSum { get; set; }
        public IActivationFunction ActivationFunction { get; set; }

        public Node() 
        {
            this.ActivatedSum = null;
        }

        public Node(IActivationFunction activationFunction)
        {
            this.ActivatedSum = null;
            this.ActivationFunction = activationFunction;
        }

        public void Activate(Layer layer = null)
        {
            if (this.ActivationFunction == null)
            {
                this.ActivatedSum = this.Input;
            }
            else
            {
                this.ActivatedSum = this.ActivationFunction.Activate(this, layer);
            }
        }
    }
}
