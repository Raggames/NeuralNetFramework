using UnityEngine;

namespace NeuralNetwork.Scripts.Controllers
{
   
    public class JumperController : NeuralNetController
    {
        public float DistanceToUp;
        public float DistanceToDown;
        public float DistanceToRight;
        
        public float JumpForce;
        
        public Rigidbody Rigidbody;

        public float Timer;
        private Vector3 startPos;

        [SerializeField] private bool isDead;
        private void Start()
        {
            Debug.Log("Controller Start");
            Rigidbody = GetComponent<Rigidbody>();
            startPos = transform.position;
        }

        public override void SetInputs()
        {
            if (NeuralNet.IsTraining || NeuralNet.IsExecuting)
            {
                NeuralNet.ExternalInputs[0] = DistanceToUp;
                NeuralNet.ExternalInputs[1] = DistanceToDown;
            }
        }
        public override void OnOutput()
        {
            if (NeuralNet.OutputToExternal[0] > 0.1)
            {
                Rigidbody.AddForce(Vector3.up*JumpForce);
            }
            JumpForce = (float)NeuralNet.OutputToExternal[1];
        }

        public override void OnInstanceFail()
        {
            if (!isDead)
            {
                EvaluationParameters[0].EvaluationParameter = Timer;
                //EvaluationParameters[1] = JumpForce;
                this.NeuralNet.Genetic_OnInstanceEnd(EvaluationParameters);
                isDead = true;
                gameObject.SetActive(false);
            }
            
        }
        public override void InstanceReset()
        {
            Debug.Log("Instance Reset");
            isDead = false;
            Timer = 0;
            JumpForce = 0;
            transform.position = startPos;
            Rigidbody.velocity = Vector3.zero;
        }


        private void Update()
        {
            if (NeuralNet.inputStreamOn && !isDead)
            {
                if (transform.position.y > 9.4)
                {
                    OnInstanceFail();   
                }

                if (transform.position.y < 0.6)
                {
                    OnInstanceFail(); 
                }
               
                DistanceToUp = Vector3.Distance(transform.position, new Vector3(transform.position.x,10,transform.position.z));
                DistanceToDown = Vector3.Distance(transform.position, new Vector3(transform.position.x,0,transform.position.z));
                SetInputs();
                NeuralNet.UseInstance(NeuralNet.ExternalInputs);
                OnOutput();
                Timer += Time.deltaTime;
            }
           
            
            
        }
    }
}