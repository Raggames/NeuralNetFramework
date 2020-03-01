using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NeuralNetwork.Scripts.Controllers
{
    public class ShooterController : NeuralNetController
    {
        
        [Header("Inputs")]
        
        public List<Transform> enemies = new List<Transform>();
       
        public float FireTimer;
        public Vector2 SpawnZone = new Vector2();        
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

        private bool lastTankAlive;

        public bool combo;
        // Start is called before the first frame update
        void Start()
        {
            Rigidbody = GetComponent<Rigidbody>();
            RandomPosition();
            startPos = transform.position;
            Timer = 0;
            Debug.Log("Start");
            
        }

        void RandomPosition()
        {
            Vector3 RandomPos = new Vector3();
            RandomPos.x = Random.Range(-SpawnZone.x, SpawnZone.x);
            RandomPos.z = Random.Range(-SpawnZone.y, SpawnZone.y);
            RandomPos.y = 0.6f;
            transform.position = RandomPos;
        }
        // Update is called once per frame
        void Update()
        {
            if (NeuralNet.inputStreamOn && !isDead)
            {
                enemies.Clear();
                Collider[] hitColliders = Physics.OverlapSphere(transform.position, 40);
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
                    lastTankAlive = true;
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
                    lastTankAlive = true;
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
                NeuralNet.ExternalInputs[0] = enemies[0].position.x;
                NeuralNet.ExternalInputs[1] = enemies[0].position.z;
            }
            else
            {
                NeuralNet.ExternalInputs[0] = 0;
                NeuralNet.ExternalInputs[1] = 0;
            }
            NeuralNet.ExternalInputs[2] = FireTimer;
            NeuralNet.ExternalInputs[3] = transform.position.x;
            NeuralNet.ExternalInputs[4] = transform.position.z;
        }

        public override void OnOutput()
        {
            
            if (NeuralNet.OutputToExternal[0] < 0)
            {
                MoveLeft();
            }
            if (NeuralNet.OutputToExternal[0] > 0)
            {
                MoveRight();
            }
            
            if (NeuralNet.OutputToExternal[1] > 0)
            {
                MoveForward();   
            }
            if (NeuralNet.OutputToExternal[1] < 0)
            {
                MoveBackward();
            }
            if (NeuralNet.OutputToExternal[2] >= .5)
            {
                Aim();
            }

            if (NeuralNet.OutputToExternal[3] > 0)
            {
                Shoot();
            }
        }

        private void MoveLeft()
        {
            Rigidbody.AddForce(Vector3.left*MoveForce);
        }
        private void MoveRight()
        {
            Rigidbody.AddForce(Vector3.right*MoveForce);
        }

        private void MoveForward()
        {
            Rigidbody.AddForce(Vector3.forward*MoveForce);
        }

        private void MoveBackward()
        {
            Rigidbody.AddForce(Vector3.back*MoveForce);
        }

        private void RotateTurretRight()
        {
            Turret.Rotate(Vector3.up, RotationSpeed*Time.deltaTime);
        }

        private void RotateTurretLeft()
        {
            Turret.Rotate(Vector3.up, -RotationSpeed*Time.deltaTime);

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
                EvaluationParameters[0].EvaluationParameter = Timer;
                EvaluationParameters[1].EvaluationParameter = Points;
                this.NeuralNet.Genetic_OnInstanceEnd(EvaluationParameters);
                isDead = true;
                gameObject.SetActive(false);
            }
        }
        public override void InstanceReset()
        {
            isDead = false;
            RandomPosition();
            lastTankAlive = false;
            Timer = 0;
            combo = false;
            Points = 0;
        }
    }

}
