using UnityEngine;

namespace DefaultNamespace
{
    public class JumperController : NeuralNetworkController
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
            Rigidbody = GetComponent<Rigidbody>();
            startPos = transform.position;
        }

        public override void SetInputs()
        {
            NeuralNetworkComponent.NetInput[0].InputValue = DistanceToUp;
            NeuralNetworkComponent.NetInput[1].InputValue = DistanceToDown;
            
        }
        public override void OnOutput()
        {
            if (NeuralNetworkComponent.NetOutput[0].OutputValue > 0.1)
            {
                Rigidbody.AddForce(Vector3.up*JumpForce);
            }
           
            JumpForce = (float)NeuralNetworkComponent.NetOutput[1].OutputValue;
        }

        public override void OnInstanceFail()
        {
            if (!isDead)
            {
                EvaluationParameters[0] = Timer;
                //EvaluationParameters[1] = JumpForce;
                this.NeuralNetworkComponent.InstanceEnd(EvaluationParameters, this);
                isDead = true;
                gameObject.SetActive(false);
            }
            
        }
        public override void InstanceReset()
        {
            isDead = false;
            Timer = 0;
            JumpForce = (float)NeuralNetworkComponent.NetOutput[1].OutputValue;
            EvaluationParameters[0] = Timer;
            transform.position = startPos;
            Rigidbody.velocity = Vector3.zero;

        }


        private void Update()
        {
            if (NeuralNetworkComponent.inputStreamOn && !isDead)
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
                OnOutput();
                Timer += Time.deltaTime;
            }
           
            
            
        }
    }
}