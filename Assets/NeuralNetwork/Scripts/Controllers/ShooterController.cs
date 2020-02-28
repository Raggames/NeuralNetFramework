using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NeuralNetwork.Scripts.Controllers
{
    public class ShooterController : NeuralNetworkController
    {
        
        [Header("Inputs")]
        
        public List<Transform> enemies = new List<Transform>();
       
        public float FireTimer;
        
        [Header("Management")]
        public float FireRate;
        
        public Rigidbody Rigidbody;

        public float Timer;
        private Vector3 startPos;
        public float MoveForce;
        public float ShootForce;
        public float RotationSpeed = 3;
        public GameObject Bullet;
        public Transform Turret;
        public Transform GunEnd;
        [SerializeField] private bool isDead;
        public int Points;
        

        // Start is called before the first frame update
        void Start()
        {
            Rigidbody = GetComponent<Rigidbody>();
            RandomPosition();
            startPos = transform.position;
            Timer = 30;
            Debug.Log("Start");
            
        }

        void RandomPosition()
        {
            Vector3 RandomPos = new Vector3();
            RandomPos.x = Random.Range(-9, 9);
            RandomPos.z = Random.Range(-9, 9);
            RandomPos.y = 0.6f;
            transform.position = RandomPos;
        }
        // Update is called once per frame
        void Update()
        {
            if (NeuralNetworkComponent.inputStreamOn && !isDead)
            {
                enemies.Clear();
                Collider[] hitColliders = Physics.OverlapSphere(transform.position, 20);
                int i = 0;
                while (i < hitColliders.Length)
                {
                    if (hitColliders[i].CompareTag("Tank"))
                    {
                        if (hitColliders[i].gameObject != gameObject)
                        {
                            enemies.Add(hitColliders[i].gameObject.transform);

                        }
                    }
                    i++;
                }

                if (enemies.Count == 0)
                {
                    Debug.Log("Last survivor");
                    OnInstanceFail();
                }
                enemies.Sort(delegate(Transform transform1, Transform transform2)
                    {
                        return Vector3.Distance(transform1.position, transform.position)
                            .CompareTo(Vector3.Distance(transform2.position, transform.position));
                        
                    });
                
                FireTimer += Time.deltaTime;
                Timer += Time.deltaTime;
                if (Timer >= 30)
                {
                    OnInstanceFail();
                }
                SetInputs();
                OnOutput();
            }
        }

        public override void SetInputs()
        {
            if (enemies.Count > 0)
            {
                NeuralNetworkComponent.NetInput[0].InputValue = enemies[0].position.x;
                NeuralNetworkComponent.NetInput[1].InputValue = enemies[0].position.z;
            }
            else
            {
                NeuralNetworkComponent.NetInput[0].InputValue = 0;
            }
          
           NeuralNetworkComponent.NetInput[2].InputValue = FireTimer;
        }

        public override void OnOutput()
        {
            
            if (NeuralNetworkComponent.NetOutput[0].OutputValue < 0.4f)
            {
                MoveLeft();
            }
            if (NeuralNetworkComponent.NetOutput[0].OutputValue > 0.6f)
            {
                MoveRight();
            }
            
            if (NeuralNetworkComponent.NetOutput[1].OutputValue > 0.6f)
            {
                MoveForward();   
            }
            if (NeuralNetworkComponent.NetOutput[1].OutputValue < 0.4f)
            {
                MoveBackward();
            }
            if (NeuralNetworkComponent.NetOutput[2].OutputValue >= .5)
            {
                Aim();
            }

            if (NeuralNetworkComponent.NetOutput[3].OutputValue >= .5)
            {
                Shoot();
            }
        }

        private void MoveLeft()
        {
            Debug.Log("Left");

            Rigidbody.AddForce(Vector3.left*MoveForce);
        }
        private void MoveRight()
        {
            Debug.Log("Right");

            Rigidbody.AddForce(Vector3.right*MoveForce);
        }

        private void MoveForward()
        {
            Debug.Log("Forward");
            Rigidbody.AddForce(Vector3.forward*MoveForce);
        }

        private void MoveBackward()
        {
            Debug.Log("Back");
            Rigidbody.AddForce(Vector3.back*MoveForce);
        }
        private void Aim()
        {
            Debug.Log("Aim");
            if (enemies.Count > 0)
            {
                Vector3 enemyPosition = enemies[0].position;
                enemyPosition.y = .8f;
                Vector3 _direction = (enemyPosition - transform.position).normalized;
 
                //create the rotation we need to be in to look at the target
                Quaternion _lookRotation = Quaternion.LookRotation(_direction);
 
                //rotate us over time according to speed until we are in the required rotation
                Turret.transform.rotation = _lookRotation;
            }
        }
        private void Shoot()
        {
            Debug.Log("Shoot");

            if (FireTimer >= FireRate)
            {
                var Bullet = Instantiate(this.Bullet, GunEnd.position, GunEnd.rotation);
                Bullet.GetComponent<BulletComponent>().controller = this;
                Bullet.GetComponent<Rigidbody>().AddForce(Bullet.transform.TransformDirection(Vector3.forward*ShootForce), ForceMode.VelocityChange);
                FireTimer = 0;
            }
        }

        public override void OnInstanceFail()
        {
            if (!isDead)
            {
                EvaluationParameters[0] = 30-Timer;
                EvaluationParameters[1] = Points;
                this.NeuralNetworkComponent.InstanceEnd(EvaluationParameters, this);
                isDead = true;
                gameObject.SetActive(false);
            }
        }

        public override void InstanceReset()
        {
            isDead = false;
            RandomPosition();
            Timer = 0;
            Points = 0;
            EvaluationParameters[1] = Points*2;
            EvaluationParameters[0] = Timer;
        }
    }

}
