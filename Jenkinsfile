pipeline {    

    agent any
    environment {
        DOCKER_HUB_REPO = 'molka11/mon-app'  
        DOCKER_HUB_CREDENTIALS = 'DockerToken' 
        SONAR_TOKEN = 'SonarToken'
        SONAR_HOST_URL = 'http://localhost:9000' 
    }
  
    stages {  

        stage('Git') {
            steps {  
                git branch: 'master', url: 'https://github.com/molka1107/projet_pfe.git', credentialsId: 'GitToken'
            }        
        }

        stage('SonarQube Analysis') {
                steps {
                    echo "Running SonarQube Analysis..."
                    withCredentials([string(credentialsId: 'SonarToken', variable: 'SONAR_TOKEN')]) {
                        withSonarQubeEnv('Sonar') {
                            sh """
                                sonar-scanner \
                                    -Dsonar.projectKey=projet_pfe \
                                    -Dsonar.sources=. \
                                    -Dsonar.host.url=$SONAR_HOST_URL \
                                    -Dsonar.login=$SONAR_TOKEN
                            """
                        }
                    }
                }
            }



       stage('Install Dependencies') {
            steps {
                script {
                    sh '''
                    bash -c "
                    python3 -m venv venv
                    source venv/bin/activate
                    pip install -r requirements.txt
                    "
                    '''
                }
            }
        }






        
        stage('Docker Build') {
            steps {
                withCredentials([usernamePassword(credentialsId: "${DOCKER_HUB_CREDENTIALS}", passwordVariable: 'DOCKER_PASSWORD', usernameVariable: 'DOCKER_USERNAME')]) {
                    sh 'docker build -t ${DOCKER_HUB_REPO}:latest .'

                }
            }
        }

        stage('Docker Run') {
            steps {
                sh 'docker run -d --name mon-app-streamlit -p 8501:8501 ${DOCKER_HUB_REPO}:latest'

            }
        }

        stage('Push Docker Image To Docker Hub') {
            steps {
                withCredentials([usernamePassword(credentialsId: "${DOCKER_HUB_CREDENTIALS}", passwordVariable: 'DOCKER_PASSWORD', usernameVariable: 'DOCKER_USERNAME')]) {
                    sh 'docker push ${DOCKER_HUB_REPO}:latest'
                }
            }
        }

        stage('Deploy Application') {
            steps {
                script {
                    sh 'docker compose down && docker compose up -d'
                }
            }
        }

        stage('Deploy With Docker Compose') {
            steps {
                script {
                    sh "docker compose -f docker-compose.yml up -d"
                }
            }
        }

     
      
         
     
    


        


      
        stage('Deploy to Nexus') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'NexusToken', usernameVariable: 'NEXUS_USERNAME', passwordVariable: 'NEXUS_PASSWORD')]) {
                    echo "Deploying to Nexus"
                    sh 'mvn deploy -DskipTests -DaltDeploymentRepository=deploymentRepo::default::http://localhost:8081/repository/maven-releases/ -Dusername=$NEXUS_USERNAME -Dpassword=$NEXUS_PASSWORD'
                }
            }
        }

     
       

      stage('Start Prometheus') {
            steps {
                script {
                    echo "Starting Prometheus"
                    sh "docker start prometheus"
                }
            }
        }

        stage('Start Grafana') {
            steps {
                script {
                    echo "Starting Grafana"
                    sh "docker start grafana"
                }
            }
        }

    

      
    }

}
