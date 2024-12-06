pipeline {    
    agent any
    //environment {
        //DOCKER_HUB_REPO = 'ghofranebj/stationski_backend'  // Docker Hub repository
        //DOCKER_HUB_CREDENTIALS = 'DockerToken'  // Docker Hub credentials ID in Jenkins
    //}
   
    stages {  
        stage('Checkout') {
            steps {  
                git branch: 'master', url: 'https://github.com/molka1107/projet_pfe.git', credentialsId: 'GitToken'
            }        
        }

        stage('Install Dependencies') {
            steps {
                script {
                    sh 'bash -c "source /opt/anaconda3/etc/profile.d/conda.sh && conda activate stage && pip install -r requirements.txt"'
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    sh 'pytest'
                }
            }
        }

        stage('SonarQube Analysis') {
            steps {
                withCredentials([string(credentialsId: 'SonarToken', variable: 'SONAR_TOKEN')]) {
                    withSonarQubeEnv('Sonar') {
                        sh 'mvn sonar:sonar -Dsonar.token=$SONAR_TOKEN -Dsonar.host.url=http://localhost:9000 -Dsonar.java.binaries=target/classes'
                    }
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

        stage('Docker Build') {
            steps {
                echo 'Building Docker image...'
                withCredentials([usernamePassword(credentialsId: "${DOCKER_HUB_CREDENTIALS}", passwordVariable: 'DOCKER_PASSWORD', usernameVariable: 'DOCKER_USERNAME')]) {
                    sh 'echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin'
                    sh 'docker build -t ${DOCKER_HUB_REPO}:latest .'
                }
            }
        }
stage('Docker Run') {
    steps {
        echo 'Running Docker container...'
        sh 'docker run -d -p 8089:8089 --name backend-container-${BUILD_ID} ${DOCKER_HUB_REPO}:latest'
    }
}


        stage('Push Docker Image to Docker Hub') {
            steps {
                withCredentials([usernamePassword(credentialsId: "${DOCKER_HUB_CREDENTIALS}", passwordVariable: 'DOCKER_PASSWORD', usernameVariable: 'DOCKER_USERNAME')]) {
                    sh 'echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin'
                    sh 'docker push ${DOCKER_HUB_REPO}:latest'
                }
            }
        }

      stage('Deploy with Docker Compose') {
    steps {
        script {
            echo "Starting Docker Compose"
            sh "docker-compose -f docker-compose.yml up -d" // Adjust if necessary
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
