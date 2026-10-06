//v1.3.0
#include "imder_plugins.h"
#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QFrame>
#include <QtWidgets/QSizePolicy>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QAbstractItemView>
#include <QtWidgets/QMenu>
#include <QtWidgets/QAction>
#include <QtWidgets/QSlider>
#include <QtWidgets/QColorDialog>
#include <QtWidgets/QInputDialog>
#include <QtCore/QThread>
#include <QtCore/QTimer>
#include <QtCore/QMutex>
#include <QtCore/QDateTime>
#include <QtCore/QFileInfo>
#include <QtCore/QFile>
#include <QtCore/QDir>
#include <QtCore/QTemporaryDir>
#include <QtCore/QStandardPaths>
#include <QtCore/QPoint>
#include <QtCore/QSize>
#include <QtGui/QColor>
#include <QtGui/QPalette>
#include <QtGui/QCloseEvent>
#include <QtGui/QImage>
#include <QtGui/QPixmap>
#include <QtGui/QIcon>
#include <QtGui/QPainter>
#include <QtGui/QPen>
#include <QtGui/QMouseEvent>
#include <QtGui/QStandardItemModel>
#include <QtGui/QRegion>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <numeric>
#include <random>
#include <functional>
#include <memory>
#include <sstream>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <atomic>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <cerrno>
#include <cstdlib>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <process.h>
#include <io.h>
#else
#include <unistd.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/prctl.h>
#include <signal.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void attach_parent_console(int argc,char* argv[]){
#ifdef _WIN32
    (void)argc;(void)argv;
    size_t noAttachLen=0;
    getenv_s(&noAttachLen,nullptr,0,"IMDER_NO_CONSOLE_ATTACH");
    if(noAttachLen) return;
    if(GetConsoleWindow()!=nullptr) return;
    if(!AttachConsole(ATTACH_PARENT_PROCESS)) return;
    freopen("CONOUT$","w",stdout);
    freopen("CONOUT$","w",stderr);
    freopen("CONIN$","r",stdin);
    SetConsoleOutputCP(65001);
#else
    (void)argc;(void)argv;
#endif
}

static std::string outNull(){
#ifdef _WIN32
    return " > NUL 2>&1";
#else
    return " >/dev/null 2>&1";
#endif
}

static FILE* openPipe(const std::string& cmd,const char* mode){
#ifdef _WIN32
    return _popen(cmd.c_str(),mode);
#else
    return popen(cmd.c_str(),mode[0]=='w'?"w":"r");
#endif
}

static int closePipe(FILE* p){
#ifdef _WIN32
    return _pclose(p);
#else
    return pclose(p);
#endif
}

static std::tm localTm(time_t t){
    std::tm tmv{};
#ifdef _WIN32
    localtime_s(&tmv,&t);
#else
    localtime_r(&t,&tmv);
#endif
    return tmv;
}

class Proc {
public:
    FILE* out=nullptr;
    FILE* in=nullptr;
    int exit_code=-1;

    ~Proc(){
        if(in){std::fclose(in);in=nullptr;}
        if(!reaped_) kill();
        if(out){drainOut();std::fclose(out);out=nullptr;}
        wait_close(500);
    }

    bool spawn(const std::vector<std::string>& argv,bool capture_out,bool feed_in,bool capture_err=false){
#ifdef _WIN32
        SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES),nullptr,TRUE};
        HANDLE orr=nullptr,owr=nullptr,irr=nullptr,iwr=nullptr,errr=nullptr,errw=nullptr;
        if(capture_out&&!CreatePipe(&orr,&owr,&sa,0)) return false;
        if(feed_in&&!CreatePipe(&irr,&iwr,&sa,0)) return false;
        if(capture_err&&!CreatePipe(&errr,&errw,&sa,0)) return false;
        if(orr) SetHandleInformation(orr,HANDLE_FLAG_INHERIT,0);
        if(iwr) SetHandleInformation(iwr,HANDLE_FLAG_INHERIT,0);
        if(errr) SetHandleInformation(errr,HANDLE_FLAG_INHERIT,0);
        STARTUPINFOW si{};
        si.cb=sizeof(si);
        si.dwFlags=STARTF_USESTDHANDLES;
        si.hStdInput=feed_in?irr:HANDLE(_get_osfhandle(_fileno(stdin)));
        si.hStdOutput=capture_out?owr:HANDLE(_get_osfhandle(_fileno(stdout)));
        si.hStdError=capture_err?errw:HANDLE(_get_osfhandle(_fileno(stderr)));
        std::string cmd;
        for(size_t i=0;i<argv.size();i++){
            if(i) cmd+=" ";
            cmd+=quoteWin(argv[i]);
        }
        std::wstring wcmd=wideFromUtf8(cmd);
        std::vector<wchar_t> cmdv(wcmd.begin(),wcmd.end());
        cmdv.push_back(L'\0');
        if(!CreateProcessW(nullptr,cmdv.data(),nullptr,nullptr,TRUE,CREATE_NO_WINDOW,nullptr,nullptr,&si,&pi)){
            if(orr) CloseHandle(orr);
            if(irr) CloseHandle(irr);
            if(errr) CloseHandle(errr);
            if(errw) CloseHandle(errw);
            return false;
        }
        job_=CreateJobObjectW(nullptr,nullptr);
        if(job_){
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
            li.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(job_,JobObjectExtendedLimitInformation,&li,sizeof(li));
            AssignProcessToJobObject(job_,pi.hProcess);
        }
        if(owr) CloseHandle(owr);
        if(irr) CloseHandle(irr);
        if(errw) CloseHandle(errw);
        if(capture_out) out=_fdopen(_open_osfhandle((intptr_t)orr,0),"rb");
        if(feed_in) in=_fdopen(_open_osfhandle((intptr_t)iwr,0),"wb");
        if(capture_err) start_err_thread((intptr_t)errr);
#else
        int op[2]={-1,-1},ip[2]={-1,-1},ep[2]={-1,-1};
        if(capture_out&&pipe(op)!=0) return false;
        if(feed_in&&pipe(ip)!=0) return false;
        if(capture_err&&pipe(ep)!=0) return false;
        pid=fork();
        if(pid<0) return false;
        if(pid==0){
            prctl(PR_SET_PDEATHSIG,SIGKILL);
            if(capture_out){dup2(op[1],1);close(op[0]);close(op[1]);}
            if(feed_in){dup2(ip[0],0);close(ip[0]);close(ip[1]);}
            if(capture_err){dup2(ep[1],2);close(ep[0]);close(ep[1]);}
            std::vector<char*> av;
            for(auto& a:argv) av.push_back(const_cast<char*>(a.c_str()));
            av.push_back(nullptr);
            execvp(av[0],av.data());
            _exit(127);
        }
        if(op[1]>=0) close(op[1]);
        if(ip[0]>=0) close(ip[0]);
        if(ep[1]>=0) close(ep[1]);
        if(capture_out) out=fdopen(op[0],"rb");
        if(feed_in) in=fdopen(ip[1],"wb");
        if(capture_err) start_err_thread((intptr_t)ep[0]);
#endif
        return true;
    }

    void kill(){
#ifdef _WIN32
        std::lock_guard<std::mutex> lk(lifecycle_);
        if(pi.hProcess&&!reaped_) TerminateProcess(pi.hProcess,(UINT)-1);
#else
        std::lock_guard<std::mutex> lk(lifecycle_);
        if(pid>0&&!reaped_) ::kill(pid,SIGKILL);
#endif
    }

    void abandon_out(){
        std::lock_guard<std::mutex> lk(lifecycle_);
        if(out){
            std::fclose(out);
            out=nullptr;
        }
    }

    int wait_close(int timeout_ms=-1){
        {
            std::lock_guard<std::mutex> lk(lifecycle_);
            if(in){std::fclose(in);in=nullptr;}
            if(out){drainOut();std::fclose(out);out=nullptr;}
        }
#ifdef _WIN32
        HANDLE h=nullptr;
        {
            std::lock_guard<std::mutex> lk(lifecycle_);
            if(!reaped_&&pi.hProcess) h=pi.hProcess;
        }
        if(h){
            DWORD w=WaitForSingleObject(h,timeout_ms<0?INFINITE:DWORD(timeout_ms));
            if(w!=WAIT_OBJECT_0) TerminateProcess(h,(UINT)-1);
            WaitForSingleObject(h,INFINITE);
            std::lock_guard<std::mutex> lk(lifecycle_);
            if(!reaped_){
                DWORD code=0;
                GetExitCodeProcess(h,&code);
                exit_code=(int)code;
                if(job_) CloseHandle(job_),job_=nullptr;
                CloseHandle(pi.hProcess);
                CloseHandle(pi.hThread);
                pi=PROCESS_INFORMATION{};
                reaped_=true;
            }
        }
#else
        pid_t target=-1;
        {
            std::lock_guard<std::mutex> lk(lifecycle_);
            if(!reaped_&&pid>0) target=pid;
        }
        if(target>0){
            auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms<0?3600000:timeout_ms);
            int st=0;
            while(waitpid(target,&st,WNOHANG)==0){
                if(std::chrono::steady_clock::now()>deadline){
                    ::kill(target,SIGKILL);
                    waitpid(target,&st,0);
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(4));
            }
            std::lock_guard<std::mutex> lk(lifecycle_);
            if(!reaped_){
                exit_code=WIFEXITED(st)?WEXITSTATUS(st):-1;
                pid=-1;
                reaped_=true;
            }
        }
#endif
        join_err_thread();
        return exit_code;
    }

    std::string stderr_tail(){
        std::lock_guard<std::mutex> lk(err_mutex_);
        return err_tail_;
    }

private:
#ifdef _WIN32
    PROCESS_INFORMATION pi{};
    HANDLE job_=nullptr;
    static std::string quoteWin(const std::string& s){
        if(s.find(' ')==std::string::npos&&!s.empty()) return s;
        return '"'+s+'"';
    }
    static std::wstring wideFromUtf8(const std::string& s){
        if(s.empty()) return std::wstring();
        int n=MultiByteToWideChar(CP_UTF8,0,s.c_str(),(int)s.size(),nullptr,0);
        std::wstring w(n,L'\0');
        MultiByteToWideChar(CP_UTF8,0,s.c_str(),(int)s.size(),&w[0],n);
        return w;
    }
#else
    pid_t pid=-1;
#endif
    std::mutex lifecycle_;
    std::mutex err_mutex_;
    std::thread err_th_;
    std::string err_tail_;
    bool reaped_=false;

    void drainOut(){
        if(!out) return;
        char buf[4096];
        while(std::fread(buf,1,sizeof(buf),out)>0){}
    }

    void start_err_thread(intptr_t handle){
        err_th_=std::thread([this,handle]{
            std::string acc;
#ifdef _WIN32
            HANDLE h=(HANDLE)handle;
            char buf[4096];
            DWORD nread=0;
            while(ReadFile(h,buf,sizeof(buf),&nread,nullptr)&&nread){
                acc.append(buf,nread);
                if(acc.size()>8192) acc.erase(0,acc.size()-8192);
            }
            CloseHandle(h);
#else
            int fd=(int)handle;
            char buf[4096];
            ssize_t nread=0;
            while((nread=read(fd,buf,sizeof(buf)))>0){
                acc.append(buf,(size_t)nread);
                if(acc.size()>8192) acc.erase(0,acc.size()-8192);
            }
            close(fd);
#endif
            std::lock_guard<std::mutex> lk(err_mutex_);
            err_tail_=acc;
        });
    }

    void join_err_thread(){
        if(err_th_.joinable()) err_th_.join();
    }
};

static bool checkFfmpeg(){FILE* p=openPipe("ffmpeg -version","rb");if(!p){fprintf(stderr,"[imder] ffmpeg probe: pipe failed (errno %d)\n",errno);return false;}char buf[256];size_t n=fread(buf,1,sizeof(buf),p);int rc=closePipe(p);if(n==0)fprintf(stderr,"[imder] ffmpeg probe: no output (exit %d)\n",rc);return n>0;}

struct VideoInfo {
    bool ok=false;
    int w=0,h=0;
    double fps=30.0;
    double dur=0.0;
    int frames=0;
};

static VideoInfo probeVideoInfo(const std::string& path){
    VideoInfo vi;
    std::string cmd="ffprobe -v quiet -select_streams v:0 -show_entries stream=width,height,r_frame_rate -show_entries format=duration -of csv=p=0 \""+path+"\"";
    FILE* pr=openPipe(cmd.c_str(),"rb");
    if(!pr) return vi;
    char buf[1024]={0};
    size_t n=fread(buf,1,sizeof(buf)-1,pr);
    closePipe(pr);
    if(n==0) return vi;
    double dur=0.0;
    char* nl=strchr(buf,'\n');
    if(nl){
        *nl=0;
        dur=atof(nl+1);
    }
    int iw=0,ih=0;
    char rate[64]={0};
    if(sscanf(buf,"%d,%d,%63s",&iw,&ih,rate)!=3) return vi;
    int num=0,den=1;
    sscanf(rate,"%d/%d",&num,&den);
    if(iw<=0||ih<=0) return vi;
    double fps=(num>0&&den>0)?(double)num/(double)den:30.0;
    if(fps<=0||fps>1000) fps=30.0;
    vi.ok=true;
    vi.w=iw;vi.h=ih;vi.fps=fps;
    vi.dur=dur>0?dur:0.0;
    vi.frames=dur>0?(int)(dur*fps+0.5):0;
    return vi;
}

struct FfmpegWriter {
    FILE* p=nullptr;
    bool open(const std::string& path,int w,int h,double fps){
        if(!checkFfmpeg()) return false;
        std::string cmd="ffmpeg -y -f rawvideo -pix_fmt bgr24 -s "+std::to_string(w)+"x"+std::to_string(h)+
        " -r "+std::to_string(fps)+" -i - -c:v mpeg4 -q:v 2 \""+path+"\""+outNull();
        p=openPipe(cmd.c_str(),"wb");
        if(!p) fprintf(stderr,"[imder] video pipe failed (errno %d): %s\n",errno,cmd.c_str());
        return p!=nullptr;
    }
    bool isOpened() const {return p!=nullptr;}
    bool warned=false;
    void write(const cv::Mat& bgr){ if(!p) return; size_t tot=(size_t)bgr.total()*bgr.elemSize(); if(fwrite(bgr.data,1,tot,p)!=tot&&!warned){warned=true;fprintf(stderr,"[imder] video frame write failed - ffmpeg exited early\n");} }
    void release(){ if(p){ closePipe(p); p=nullptr; } }
};

struct FfmpegReader {
    FILE* p=nullptr;
    int w=0,h=0;
    double fps=30.0;
    bool open(const std::string& path){
        if(!probeMeta(path)) return false;
        std::string cmd="ffmpeg -v quiet -i \""+path+"\" -f rawvideo -pix_fmt bgr24 -";
        p=openPipe(cmd.c_str(),"rb");
        return p!=nullptr;
    }
    bool probeMeta(const std::string& path){
        std::string cmd="ffprobe -v quiet -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of csv=p=0 \""+path+"\"";
        FILE* pr=openPipe(cmd.c_str(),"rb");
        if(!pr) return false;
        char buf[512]={0}; size_t n=fread(buf,1,sizeof(buf)-1,pr); closePipe(pr);
        if(n==0) return false;
        int iw=0,ih=0; char rate[64]={0};
        if(sscanf(buf,"%d,%d,%63s",&iw,&ih,rate)!=3) return false;
        int num=0,den=1; sscanf(rate,"%d/%d",&num,&den);
        if(iw<=0||ih<=0) return false;
        if(num>0&&den>0) fps=(double)num/(double)den;
        if(fps<=0||fps>1000) fps=30.0;
        w=iw;h=ih;
        return true;
    }
    bool read(cv::Mat& out){
        if(!p) return false;
        cv::Mat frm(h,w,CV_8UC3);
        size_t got=fread(frm.data,1,(size_t)w*h*3,p);
        if(got!=(size_t)w*h*3) return false;
        out=frm; return true;
    }
    void release(){ if(p){ closePipe(p); p=nullptr; } }
};

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static cv::Mat decodeWithStb(const std::string& path){
    FILE* f=fopen(path.c_str(),"rb");
    if(!f) return cv::Mat();
    fseek(f,0,SEEK_END);
    long sz=ftell(f);
    fseek(f,0,SEEK_SET);
    if(sz<=0){fclose(f);return cv::Mat();}
    std::vector<uint8_t> buf((size_t)sz);
    if(fread(buf.data(),1,(size_t)sz,f)!=(size_t)sz){fclose(f);return cv::Mat();}
    fclose(f);
    int w=0,h=0,c=0;
    stbi_uc* px=stbi_load_from_memory(buf.data(),(int)buf.size(),&w,&h,&c,3);
    if(!px) return cv::Mat();
    cv::Mat rgb(h,w,CV_8UC3,px);
    cv::Mat own=rgb.clone();
    stbi_image_free(px);
    cv::Mat bgr;
    cv::cvtColor(own,bgr,cv::COLOR_RGB2BGR);
    return bgr;
}

static cv::Mat decodeWithFfmpeg(const std::string& path){
    VideoInfo vi=probeVideoInfo(path);
    if(!vi.ok) return cv::Mat();
    std::string cmd="ffmpeg -v quiet -i \""+path+"\" -frames:v 1 -f rawvideo -pix_fmt bgr24 -";
    FILE* p=openPipe(cmd.c_str(),"rb");
    if(!p) return cv::Mat();
    cv::Mat m(vi.h,vi.w,CV_8UC3);
    size_t need=(size_t)vi.w*vi.h*3;
    size_t got=fread(m.data,1,need,p);
    closePipe(p);
    if(got!=need) return cv::Mat();
    return m;
}

static cv::Mat readImageSafe(const std::string& path){
    cv::Mat m;
    try{ m=cv::imread(path); }catch(const cv::Exception&){ m=cv::Mat(); }
    if(!m.empty()) return m;
    m=decodeWithStb(path);
    if(!m.empty()) return m;
    return decodeWithFfmpeg(path);
}

static bool writeImageSafe(const std::string& path,const cv::Mat& bgr){
    try{ if(cv::imwrite(path,bgr)) return true; }catch(const cv::Exception&){}
    std::string png=path;
    size_t d=png.rfind('.');
    if(d!=std::string::npos) png=png.substr(0,d)+".png";
    else png+=".png";
    if(png!=path){
        try{ if(cv::imwrite(png,bgr)) return true; }catch(const cv::Exception&){}
    }
    return false;
}

static QString mainBtnStyle(){return R"(
    QPushButton{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 #121212,stop:0.3 #121212,stop:0.7 #1a1a1a,stop:1 #121212);
        border:2px solid #E5E5E5;border-radius:8px;font-size:14px;
        font-weight:bold;color:white;padding:8px 16px;}
    QPushButton:hover{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 #121212,stop:0.3 #161616,stop:0.7 #1e1e1e,stop:1 #121212);
        border:2px solid #E5E5E5;}
    QPushButton:pressed{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 #0e0e0e,stop:0.3 #121212,stop:0.7 #161616,stop:1 #0e0e0e);
        border:2px solid #E5E5E5;}
    QPushButton:disabled{background-color:#2a2a2a;border:2px solid #555;color:#666;})"; }

static QString surfBtnStyle(){return R"(
    QPushButton{background-color:#2a2a2a;color:white;border:1px solid #3a3a3a;
        border-radius:5px;font-size:12px;padding:6px 12px;}
    QPushButton:hover{background-color:#3a3a3a;border:1px solid #E5E5E5;}
    QPushButton:pressed{background-color:#4a4a4a;border:1px solid #E5E5E5;}
    QPushButton:disabled{background-color:#2a2a2a;border:1px solid #404040;color:#666;})"; }

static QString panelStyle(){return
    "QFrame{background-color:#121212;border:2px solid #E5E5E5;border-radius:8px;}";}

static QString comboStyle(){return R"(
    QComboBox{background-color:#1a1a1a;color:#E5E5E5;border:2px solid #E5E5E5;
        border-radius:6px;padding:6px 12px;min-width:120px;font-size:13px;}
    QComboBox:hover{border:2px solid #E5E5E5;}
    QComboBox:disabled{background-color:#2a2a2a;border:2px solid #555;color:#666;}
    QComboBox::drop-down{border:none;width:24px;}
    QComboBox QAbstractItemView{background-color:#1a1a1a;color:#E5E5E5;
        border:1px solid #E5E5E5;selection-background-color:#2a2a2a;
        outline:none;border-radius:0;}
    QComboBox QAbstractItemView::item{min-height:24px;padding:2px 8px;
        border:none;border-radius:0;}
    QComboBox QAbstractItemView::item:selected{background-color:#2a2a2a;})"; }

static QString progressStyle(){return R"(
    QProgressBar{border:1px solid #404040;background-color:#1a1a1a;height:8px;
        border-radius:4px;text-align:center;color:#A0A0A0;}
    QProgressBar::chunk{background-color:#A0A0A0;border-radius:3px;})"; }

static QString menuStyle(){return R"(
    QMenu{background-color:#1a1a1a;color:#E5E5E5;border:1px solid #E5E5E5;
        border-radius:0;padding:4px;}
    QMenu::item{padding:6px 16px;border-radius:2px;}
    QMenu::item:selected{background-color:#2a2a2a;}
    QMenu::item:disabled{color:#666;}
    QMenu::separator{height:1px;background:#3a3a3a;margin:4px 8px;})"; }

static QString sliderStyle(){return R"(
    QSlider::groove:horizontal{height:6px;background:#2a2a2a;border-radius:3px;}
    QSlider::handle:horizontal{background:#4CAF50;width:16px;height:16px;
        margin:-5px 0;border-radius:8px;})"; }

static QString titleLblStyle() { return "color:#E5E5E5;font-weight:bold;font-size:16px;"; }
static QString subtitleLblStyle(){ return "color:#A0A0A0;font-size:12px;"; }
static QString windowStyle()    { return "background-color:#0A0A0A;color:#E5E5E5;"; }
static QString previewLblStyle(){ return "background-color:#1a1a1a;border:1px solid #404040;border-radius:4px;"; }

static void applyDarkPalette(){
    QPalette pal;
    pal.setColor(QPalette::Window,QColor(0x0a,0x0a,0x0a));
    pal.setColor(QPalette::WindowText,QColor(0xe5,0xe5,0xe5));
    pal.setColor(QPalette::Base,QColor(0x1a,0x1a,0x1a));
    pal.setColor(QPalette::AlternateBase,QColor(0x22,0x22,0x22));
    pal.setColor(QPalette::Text,QColor(0xe5,0xe5,0xe5));
    pal.setColor(QPalette::Button,QColor(0x1a,0x1a,0x1a));
    pal.setColor(QPalette::ButtonText,QColor(0xe5,0xe5,0xe5));
    pal.setColor(QPalette::Highlight,QColor(0x2a,0x2a,0x2a));
    pal.setColor(QPalette::HighlightedText,QColor(0xff,0xff,0xff));
    pal.setColor(QPalette::ToolTipBase,QColor(0x1a,0x1a,0x1a));
    pal.setColor(QPalette::ToolTipText,QColor(0xe5,0xe5,0xe5));
    pal.setColor(QPalette::PlaceholderText,QColor(0xa0,0xa0,0xa0));
    pal.setColor(QPalette::Disabled,QPalette::WindowText,QColor(0x66,0x66,0x66));
    pal.setColor(QPalette::Disabled,QPalette::Text,QColor(0x66,0x66,0x66));
    pal.setColor(QPalette::Disabled,QPalette::Button,QColor(0x2a,0x2a,0x2a));
    QApplication::setPalette(pal);
}

static void tuneComboPopup(QComboBox* c){
    c->view()->setFrameShape(QFrame::NoFrame);
}

static void tuneMenu(QMenu* m){
    m->setWindowFlags(m->windowFlags()|Qt::NoDropShadowWindowHint);
}

class PaintedProgressBar : public QProgressBar {
    Q_OBJECT
public:
    QString stage;
    explicit PaintedProgressBar(QWidget* p=nullptr):QProgressBar(p){ setTextVisible(false); setFixedHeight(22); }
protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        QRectF r=rect();
        p.setPen(QPen(QColor(0x40,0x40,0x40),1));
        p.setBrush(QColor(0x1a,0x1a,0x1a));
        p.drawRoundedRect(r.adjusted(0.5,0.5,-0.5,-0.5),4,4);
        double pct=(maximum()>minimum())?(double)(value()-minimum())/(double)(maximum()-minimum()):0.0;
        QRectF cr=r.adjusted(1,1,-1,-1);
        double fw=cr.width()*pct;
        QRectF fill;
        if(fw>0.5) fill=QRectF(cr.left(),cr.top(),fw,cr.height());
        if(!fill.isEmpty()){
            p.setPen(Qt::NoPen);
            p.setBrush(QColor(0xa0,0xa0,0xa0));
            p.drawRoundedRect(fill,3,3);
        }
        QString txt=stage.isEmpty()?QString("%1%").arg(value()):QString("%1%  %2").arg(value()).arg(stage);
        p.setFont(font());
        if(!fill.isEmpty()){
            p.setClipRect(fill);
            p.setPen(QColor(0x11,0x11,0x11));
            p.drawText(r,Qt::AlignCenter,txt);
        }
        p.setClipRegion(QRegion(r.toRect()).subtracted(QRegion(fill.toRect())));
        p.setPen(QColor(0xa0,0xa0,0xa0));
        p.drawText(r,Qt::AlignCenter,txt);
    }
};

static const uint32_t SHA_K[64]={
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

static void sha256Block(uint32_t s[8],const uint8_t b[64]){
    uint32_t w[64],a,b2,c,d,e,f,g,h,t1,t2;
    for(int i=0;i<16;i++) w[i]=((uint32_t)b[i*4]<<24)|((uint32_t)b[i*4+1]<<16)|((uint32_t)b[i*4+2]<<8)|b[i*4+3];
    for(int i=16;i<64;i++){
        uint32_t s0=(w[i-15]>>7|w[i-15]<<25)^(w[i-15]>>18|w[i-15]<<14)^(w[i-15]>>3);
        uint32_t s1=(w[i-2]>>17|w[i-2]<<15)^(w[i-2]>>19|w[i-2]<<13)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    a=s[0];b2=s[1];c=s[2];d=s[3];e=s[4];f=s[5];g=s[6];h=s[7];
    for(int i=0;i<64;i++){
        uint32_t S1=(e>>6|e<<26)^(e>>11|e<<21)^(e>>25|e<<7);
        t1=h+S1+((e&f)^(~e&g))+SHA_K[i]+w[i];
        uint32_t S0=(a>>2|a<<30)^(a>>13|a<<19)^(a>>22|a<<10);
        t2=S0+((a&b2)^(a&c)^(b2&c));
        h=g;g=f;f=e;e=d+t1;d=c;c=b2;b2=a;a=t1+t2;
    }
    s[0]+=a;s[1]+=b2;s[2]+=c;s[3]+=d;s[4]+=e;s[5]+=f;s[6]+=g;s[7]+=h;
}

static std::string sha256hex(const uint8_t* data,size_t len){
    uint32_t s[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint8_t blk[64]; size_t i=0;
    while(i+64<=len){sha256Block(s,data+i);i+=64;}
    size_t rem=len-i; memcpy(blk,data+i,rem); blk[rem]=0x80;
    if(rem<56) memset(blk+rem+1,0,55-rem);
    else{memset(blk+rem+1,0,63-rem);sha256Block(s,blk);memset(blk,0,56);}
    uint64_t bits=(uint64_t)len*8;
    for(int j=0;j<8;j++) blk[56+j]=(uint8_t)(bits>>(56-j*8));
    sha256Block(s,blk);
    char hex[65]; for(int j=0;j<32;j++) snprintf(hex+j*2,3,"%02x",(s[j/4]>>(24-(j%4)*8))&0xFF);
    return std::string(hex,64);
}

static bool writeWav(const std::string& path,const std::vector<int16_t>& s,int sr=44100){
    FILE* f=fopen(path.c_str(),"wb"); if(!f) return false;
    uint32_t ds=(uint32_t)(s.size()*2),rs=36+ds;
    fwrite("RIFF",1,4,f);fwrite(&rs,4,1,f);fwrite("WAVE",1,4,f);
    fwrite("fmt ",1,4,f);uint32_t fs=16;fwrite(&fs,4,1,f);
    uint16_t af=1,ch=1;fwrite(&af,2,1,f);fwrite(&ch,2,1,f);
    uint32_t sr2=(uint32_t)sr;fwrite(&sr2,4,1,f);
    uint32_t br=(uint32_t)sr*2;fwrite(&br,4,1,f);
    uint16_t ba=2,bps=16;fwrite(&ba,2,1,f);fwrite(&bps,2,1,f);
    fwrite("data",1,4,f);fwrite(&ds,4,1,f);fwrite(s.data(),2,s.size(),f);
    fclose(f);return true;
}

static bool extractAudio(const std::string& vid,const std::string& out,double dur,int quality,bool isHz=false){
    if(!checkFfmpeg()){fprintf(stderr,"Error: ffmpeg not found.\n");return false;}
    const char* qmap[]={"32k","64k","96k","128k","160k","192k","224k","256k","320k","copy"};
    std::string cmd="ffmpeg -i \""+vid+"\"";
    if(dur>0) cmd+=" -t "+std::to_string(dur);
    if(isHz) cmd+=" -ar "+std::to_string(quality);
    else{
        int qi=quality/10-1; if(qi<0)qi=0; if(qi>9)qi=9;
        if(std::string(qmap[qi])=="copy") cmd+=" -c:a copy";
        else cmd+=" -b:a "+std::string(qmap[qi]);
    }
    cmd+=" -y \""+out+"\""+outNull();
    return system(cmd.c_str())==0;
}

static void genSoundForFrame(const cv::Mat& rgbFrame,double frameDur,
                             int sampleRate,std::vector<int16_t>& out){
    std::string hex=sha256hex(rgbFrame.data,(size_t)rgbFrame.total()*3);
    float freqs[3],amps[3];
    for(int i=0;i<3;i++){
        unsigned fs=0,as=0;
        sscanf(hex.c_str()+i*4,"%4x",&fs);
        sscanf(hex.c_str()+(i+3)*4,"%4x",&as);
        freqs[i]=50.0f+(float)(fs%4000);
        amps[i] =0.1f +(float)(as%9000)/10000.0f;
    }
    int n=(int)(sampleRate*frameDur);
    std::vector<float> wave(n,0.f);
    for(int i=0;i<3;i++)
        for(int k=0;k<n;k++)
            wave[k]+=amps[i]*sinf(2.f*(float)M_PI*freqs[i]*k/(float)sampleRate);
    float mx=0.f; for(float v:wave) mx=std::max(mx,fabsf(v));
    for(int k=0;k<n;k++){
        float v=mx>0?wave[k]/mx:wave[k];
        out.push_back((int16_t)(v*32767.f));
    }
}

static std::string addAudioToVideo(const std::string& videoPath,
                                   const std::vector<cv::Mat>& frames,
                                   double fps,const std::string& outPath,
                                   const std::string& soundOpt,
                                   const std::string& targetAudioPath,
                                   int audioQuality,bool audioHz=false){
    if(soundOpt=="mute") return videoPath;
    if(!checkFfmpeg()){fprintf(stderr,"Error: ffmpeg not found.\n");return videoPath;}

    QTemporaryDir tmpDir;
    if(!tmpDir.isValid()) return videoPath;
    std::string tmpPath=tmpDir.path().toStdString();
    std::string audioPath=tmpPath+"/audio.mp3";

    if(soundOpt=="target-sound"&&!targetAudioPath.empty()){
        double dur=frames.empty()?0.0:(double)frames.size()/fps;
        if(!extractAudio(targetAudioPath,audioPath,dur,audioQuality,audioHz)) return videoPath;
    } else if(soundOpt=="sound"){
        int sr=44100; double fd=1.0/fps;
        std::vector<int16_t> full;
        full.reserve(frames.size()*(size_t)(sr*fd+1));
        for(int i=0;i<(int)frames.size();i++){
            genSoundForFrame(frames[i],fd,sr,full);
        }
        std::string wp=tmpPath+"/audio.wav";
        audioPath=wp;
        if(!writeWav(audioPath,full,sr)) return videoPath;
    } else { return videoPath; }

    std::string cmd="ffmpeg -i \""+videoPath+"\" -i \""+audioPath+
    "\" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest -y \""+
    outPath+"\""+outNull();
    if(system(cmd.c_str())!=0){fprintf(stderr,"Error adding audio.\n");return videoPath;}
    return outPath;
}

namespace GIF {
    struct Encoder {
        std::string outputPath;
        QTemporaryDir tmpDir;
        int w, h, delayMs;
        size_t written=0;

        bool open(const std::string& path, int W, int H, int delay_ms) {
            w = W; h = H; delayMs = delay_ms;
            outputPath = path;
            written = 0;
            return tmpDir.isValid();
        }

        void writeFrame(const cv::Mat& rgb) {
            cv::Mat bgr;
            cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
            std::string framePath = tmpDir.path().toStdString() + "/frame_" + std::to_string(written) + ".png";
            if(writeImageSafe(framePath, bgr)) written++;
        }

        void close() {
            if (written == 0) return;
            double fps = 1000.0 / std::max(10, delayMs);
            std::string cmd = "ffmpeg -y -framerate " + std::to_string(fps) +
            " -i \"" + tmpDir.path().toStdString() + "/frame_%d.png\"" +
            " -vf \"scale=" + std::to_string(w) + ":" + std::to_string(h) + ":flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\"" +
            " -loop 0 \"" + outputPath + "\"" + outNull();
            system(cmd.c_str());
        }
    };
}

static inline int pyDiv(int a,int b){return a/b-(a%b!=0&&((a^b)<0)?1:0);}
static inline int pyMod(int a,int b){return a-b*pyDiv(a,b);}

static std::vector<int> argsortF(const std::vector<float>& v){
    std::vector<int> idx(v.size());std::iota(idx.begin(),idx.end(),0);
    std::sort(idx.begin(),idx.end(),[&](int a,int b){return v[a]<v[b];});
    return idx;
}

static cv::Mat applyTransforms(const cv::Mat& img,int rotSteps,bool flip){
    cv::Mat out=img.clone();
    if(flip) cv::flip(out,out,1);
    int s=rotSteps%4;
    if(s==1) cv::rotate(out,out,cv::ROTATE_90_COUNTERCLOCKWISE);
    else if(s==2) cv::rotate(out,out,cv::ROTATE_180);
    else if(s==3) cv::rotate(out,out,cv::ROTATE_90_CLOCKWISE);
    return out;
}

static cv::Mat applyUpscale(const cv::Mat& img,int tgt){
    int h=img.rows,w=img.cols,minD=std::min(h,w);
    int f=(int)((double)tgt/minD+0.9999);
    if(f>=2){cv::Mat o;cv::resize(img,o,cv::Size(w*f,h*f),0,0,cv::INTER_NEAREST);return o;}
    return img;
}

static uint32_t mortonSpread(uint32_t n){
    n=(n|(n<<16))&0x030000FF;
    n=(n|(n<< 8))&0x0300F00F;
    n=(n|(n<< 4))&0x030C30C3;
    n=(n|(n<< 2))&0x09249249;
    return n;
}
static uint32_t morton3(uint8_t r,uint8_t g,uint8_t b){
    return mortonSpread(r)|(mortonSpread(g)<<1)|(mortonSpread(b)<<2);
}

static std::vector<int> matchMorton(const std::vector<uint32_t>& sSorted,
                                    const std::vector<uint32_t>& tSorted){
    int sn=(int)sSorted.size(),tn=(int)tSorted.size();
    std::vector<int> m(tn);
    for(int i=0;i<tn;i++){
        int p=(int)(std::lower_bound(sSorted.begin(),sSorted.end(),tSorted[i])-sSorted.begin());
        m[i]=std::clamp(p,0,sn-1);
    }
    for(int i=1;i<tn;i++) if(m[i]<=m[i-1]) m[i]=m[i-1]+1;
    for(int i=0;i<tn;i++) m[i]=std::clamp(m[i],0,sn-1);
    return m;
}

static std::vector<float> quantize4(const std::vector<std::array<uint8_t,3>>& pixels){
    float fac=255.f/3.f;
    std::vector<float> r(pixels.size());
    for(int i=0;i<(int)pixels.size();i++){
        float g=0.299f*pixels[i][0]+0.587f*pixels[i][1]+0.114f*pixels[i][2];
        r[i]=std::round(g/fac)*fac;
    }
    return r;
}

static std::vector<int32_t> assignPixels(const cv::Mat& baseRGB,const cv::Mat& tgtRGB,
                                         const std::string& mode,const cv::Mat& maskFlat)
{
    int N=baseRGB.rows*baseRGB.cols;
    std::vector<int32_t> asgn;
    bool hasMask=(!maskFlat.empty()&&(int)maskFlat.total()==N);

    if(mode=="disguise") {asgn.resize(N);std::iota(asgn.begin(),asgn.end(),0);}
    else asgn.assign(N,-1);

    const uint8_t* bp=baseRGB.data;
    const uint8_t* tp=tgtRGB.data;

    if((mode=="swap"||mode=="blend")&&hasMask){
        const uint8_t* mp=maskFlat.data;
        std::vector<int> validIdx,candidIdx;
        for(int i=0;i<N;i++){if(mp[i]>0)validIdx.push_back(i);else candidIdx.push_back(i);}
        if(validIdx.empty()||candidIdx.empty()) return asgn;
        int cn=(int)candidIdx.size();
        std::vector<uint32_t> sc(cn);
        for(int i=0;i<cn;i++) sc[i]=morton3(bp[candidIdx[i]*3],bp[candidIdx[i]*3+1],bp[candidIdx[i]*3+2]);
        std::vector<int> ssIdx(cn);std::iota(ssIdx.begin(),ssIdx.end(),0);
        std::sort(ssIdx.begin(),ssIdx.end(),[&](int a,int b){return sc[a]<sc[b];});
        std::vector<uint32_t> scSorted(cn);
        for(int i=0;i<cn;i++) scSorted[i]=sc[ssIdx[i]];
        int vn=(int)validIdx.size();
        std::vector<uint32_t> tc(vn);
        for(int i=0;i<vn;i++) tc[i]=morton3(tp[validIdx[i]*3],tp[validIdx[i]*3+1],tp[validIdx[i]*3+2]);
        std::vector<int> tsIdx(vn);std::iota(tsIdx.begin(),tsIdx.end(),0);
        std::sort(tsIdx.begin(),tsIdx.end(),[&](int a,int b){return tc[a]<tc[b];});
        std::vector<uint32_t> tcSorted(vn);
        for(int i=0;i<vn;i++) tcSorted[i]=tc[tsIdx[i]];
        auto mi=matchMorton(scSorted,tcSorted);
        for(int i=0;i<vn;i++){
            int sg=candidIdx[ssIdx[mi[i]]];
            int tg=validIdx[tsIdx[i]];
            asgn[sg]=tg; asgn[tg]=sg;
        }
        return asgn;
    }

    if(mode=="navigate"&&hasMask){
        const uint8_t* mp=maskFlat.data;
        std::vector<int> validIdx;
        for(int i=0;i<N;i++) if(mp[i]>0) validIdx.push_back(i);
        if(validIdx.empty()) return asgn;
        std::vector<uint32_t> sc(N);
        for(int i=0;i<N;i++) sc[i]=morton3(bp[i*3],bp[i*3+1],bp[i*3+2]);
        std::vector<int> ssIdx(N);std::iota(ssIdx.begin(),ssIdx.end(),0);
        std::sort(ssIdx.begin(),ssIdx.end(),[&](int a,int b){return sc[a]<sc[b];});
        std::vector<uint32_t> scSorted(N);
        for(int i=0;i<N;i++) scSorted[i]=sc[ssIdx[i]];
        int vn=(int)validIdx.size();
        std::vector<uint32_t> tc(vn);
        for(int i=0;i<vn;i++) tc[i]=morton3(tp[validIdx[i]*3],tp[validIdx[i]*3+1],tp[validIdx[i]*3+2]);
        std::vector<int> tsIdx(vn);std::iota(tsIdx.begin(),tsIdx.end(),0);
        std::sort(tsIdx.begin(),tsIdx.end(),[&](int a,int b){return tc[a]<tc[b];});
        std::vector<uint32_t> tcSorted(vn);
        for(int i=0;i<vn;i++) tcSorted[i]=tc[tsIdx[i]];
        auto mi=matchMorton(scSorted,tcSorted);
        for(int i=0;i<vn;i++) asgn[ssIdx[mi[i]]]=validIdx[tsIdx[i]];
        return asgn;
    }

    if(mode=="pattern"&&hasMask){
        const uint8_t* mp=maskFlat.data;
        std::vector<int> validIdx;
        for(int i=0;i<N;i++) if(mp[i]>0) validIdx.push_back(i);
        if(validIdx.empty()) return asgn;
        int vn=(int)validIdx.size();
        std::vector<float> sg(N);
        for(int i=0;i<N;i++) sg[i]=(bp[i*3]+bp[i*3+1]+bp[i*3+2])/3.f;
        auto ssIdx=argsortF(sg);
        std::vector<std::array<uint8_t,3>> tVpx(vn);
        for(int i=0;i<vn;i++){tVpx[i]={tp[validIdx[i]*3],tp[validIdx[i]*3+1],tp[validIdx[i]*3+2]};}
        auto tq=quantize4(tVpx);
        static std::mt19937 noiseRng(12345);
        std::uniform_real_distribution<float> nd(0.f,.5f);
        for(float& v:tq) v+=nd(noiseRng);
        std::vector<int> tsIdx(vn);std::iota(tsIdx.begin(),tsIdx.end(),0);
        std::sort(tsIdx.begin(),tsIdx.end(),[&](int a,int b){return tq[a]<tq[b];});
        for(int i=0;i<vn;i++){
            float val=(vn>1)?(float)i*(N-1.f)/(vn-1.f):0.f;
            int slin=(int)val;
            asgn[ssIdx[slin]]=validIdx[tsIdx[i]];
        }
        return asgn;
    }

    if(mode=="disguise"&&hasMask){
        const uint8_t* mp=maskFlat.data;
        std::vector<int> vi; for(int i=0;i<N;i++) if(mp[i]>0) vi.push_back(i);
        if(vi.empty()) return asgn;
        int vn=(int)vi.size();
        std::vector<float> sg(vn),tg(vn);
        for(int i=0;i<vn;i++){sg[i]=(bp[vi[i]*3]+bp[vi[i]*3+1]+bp[vi[i]*3+2])/3.f;
        tg[i]=(tp[vi[i]*3]+tp[vi[i]*3+1]+tp[vi[i]*3+2])/3.f;}
        auto ss=argsortF(sg),ts=argsortF(tg);
        for(int i=0;i<vn;i++) asgn[vi[ss[i]]]=vi[ts[i]];
        return asgn;
    }

    if(mode=="fusion"&&hasMask){
        const uint8_t* mp=maskFlat.data;
        std::vector<int> vi; for(int i=0;i<N;i++) if(mp[i]>0) vi.push_back(i);
        if(vi.empty()) return asgn;
        int vn=(int)vi.size();
        std::vector<float> sg(vn),tg(vn);
        for(int i=0;i<vn;i++){sg[i]=(bp[vi[i]*3]+bp[vi[i]*3+1]+bp[vi[i]*3+2])/3.f;
        tg[i]=(tp[vi[i]*3]+tp[vi[i]*3+1]+tp[vi[i]*3+2])/3.f;}
        auto ss=argsortF(sg),ts=argsortF(tg);
        for(int i=0;i<vn;i++) asgn[vi[ss[i]]]=vi[ts[i]];
        return asgn;
    }

    if(mode=="shuffle"){
        static std::mt19937 rng(std::random_device{}());
        std::vector<int> sBlk,sWht,tBlk,tWht;
        for(int i=0;i<N;i++){
            float sg=(bp[i*3]+bp[i*3+1]+bp[i*3+2])/3.f;
            float tg=(tp[i*3]+tp[i*3+1]+tp[i*3+2])/3.f;
            if(sg>127.f) sWht.push_back(i); else sBlk.push_back(i);
            if(tg>127.f) tWht.push_back(i); else tBlk.push_back(i);
        }
        std::shuffle(sBlk.begin(),sBlk.end(),rng);std::shuffle(sWht.begin(),sWht.end(),rng);
        std::shuffle(tBlk.begin(),tBlk.end(),rng);std::shuffle(tWht.begin(),tWht.end(),rng);
        int mb=(int)std::min(sBlk.size(),tBlk.size()),mw=(int)std::min(sWht.size(),tWht.size());
        for(int i=0;i<mb;i++) asgn[sBlk[i]]=tBlk[i];
        for(int i=0;i<mw;i++) asgn[sWht[i]]=tWht[i];
        std::vector<int> sRem,tRem;
        for(int i=mb;i<(int)sBlk.size();i++) sRem.push_back(sBlk[i]);
        for(int i=mw;i<(int)sWht.size();i++) sRem.push_back(sWht[i]);
        for(int i=mb;i<(int)tBlk.size();i++) tRem.push_back(tBlk[i]);
        for(int i=mw;i<(int)tWht.size();i++) tRem.push_back(tWht[i]);
        if(!sRem.empty()&&!tRem.empty()){
            std::shuffle(sRem.begin(),sRem.end(),rng);
            std::shuffle(tRem.begin(),tRem.end(),rng);
            int mn=(int)std::min(sRem.size(),tRem.size());
            for(int i=0;i<mn;i++) asgn[sRem[i]]=tRem[i];
        }
        return asgn;
    }

    {
        std::vector<float> sg(N),tg(N);
        for(int i=0;i<N;i++){sg[i]=(bp[i*3]+bp[i*3+1]+bp[i*3+2])/3.f;
        tg[i]=(tp[i*3]+tp[i*3+1]+tp[i*3+2])/3.f;}
        auto ss=argsortF(sg),ts=argsortF(tg);
        for(int i=0;i<N;i++) asgn[ss[i]]=ts[i];
    }
    return asgn;
}

static std::vector<int32_t> assignDrawerPixels(const cv::Mat& baseRGB,const cv::Mat& tgtRGB){
    int N=baseRGB.rows*baseRGB.cols;
    std::vector<int32_t> asgn(N);std::iota(asgn.begin(),asgn.end(),0);
    const uint8_t* bp=baseRGB.data,*tp=tgtRGB.data;
    std::vector<int> nw;
    for(int i=0;i<N;i++) if(bp[i*3]<245||bp[i*3+1]<245||bp[i*3+2]<245) nw.push_back(i);
    if(nw.empty()) return asgn;
    std::vector<float> bg(nw.size());
    for(int i=0;i<(int)nw.size();i++) bg[i]=(bp[nw[i]*3]+bp[nw[i]*3+1]+bp[nw[i]*3+2])/3.f;
    auto bsort=argsortF(bg);
    std::vector<float> tg(N);
    for(int i=0;i<N;i++) tg[i]=(tp[i*3]+tp[i*3+1]+tp[i*3+2])/3.f;
    auto tsort=argsortF(tg);
    int nb=(int)nw.size();
    if(nb<=N){
        for(int i=0;i<nb;i++) asgn[nw[bsort[i]]]=tsort[std::min(i,N-1)];
    } else {
        std::set<int> used;
        for(int i=0;i<nb;i++){
            int st=std::min(i,N-1),idx=st;
            while(idx<N&&used.count(tsort[idx])) idx++;
            if(idx>=N) idx=N-1;
            used.insert(tsort[idx]);
            asgn[nw[bsort[i]]]=tsort[idx];
        }
    }
    return asgn;
}

class Missform {
public:
    int H,W,minP=0;
    std::vector<std::array<int,2>> bPos,tPos;
    std::vector<std::array<uint8_t,3>> bCol;

    Missform(const cv::Mat& bRGB,const cv::Mat& tRGB,float thr=127.f){
        H=bRGB.rows;W=bRGB.cols;
        auto mask=[&](const cv::Mat& img,std::vector<bool>& m){
            m.resize(img.rows*img.cols);
            for(int i=0;i<(int)m.size();i++) m[i]=(img.data[i*3]+img.data[i*3+1]+img.data[i*3+2])/3.f>thr;
        };
        std::vector<bool> bm,tm; mask(bRGB,bm); mask(tRGB,tm);
        for(int y=0;y<H;y++) for(int x=0;x<W;x++){
            if(bm[y*W+x]) bPos.push_back({y,x});
            if(tm[y*W+x]) tPos.push_back({y,x});
        }
        minP=(int)std::min(bPos.size(),tPos.size());
        if(minP==0) throw std::runtime_error("No valid pixels found for morphing");
        bPos.resize(minP);tPos.resize(minP);bCol.resize(minP);
        for(int i=0;i<minP;i++){
            const uint8_t* p=bRGB.data+(bPos[i][0]*W+bPos[i][1])*3;
            bCol[i]={p[0],p[1],p[2]};
        }
    }
    cv::Mat frame(float progress) const {
        cv::Mat f(H,W,CV_8UC3,cv::Scalar(0,0,0));
        if(minP==0) return f;
        float t=progress*progress*(3.f-2.f*progress);
        for(int i=0;i<minP;i++){
            int cy=(int)(bPos[i][0]+(tPos[i][0]-bPos[i][0])*t);
            int cx=(int)(bPos[i][1]+(tPos[i][1]-bPos[i][1])*t);
            if(cy>=0&&cy<H&&cx>=0&&cx<W){
                uint8_t* p=f.data+(cy*W+cx)*3;
                p[0]=bCol[i][0];p[1]=bCol[i][1];p[2]=bCol[i][2];
            }
        }
        return f;
    }
};

struct ProcessConfig {
    std::string basePath,tgtPath;
    std::string mode;
    std::string algo;
    std::string outDir;
    int   rotBase=0;  bool flipBase=false;
    int   rotTgt=0;   bool flipTgt=false;
    cv::Mat mask;
    std::vector<cv::Mat> baseShapeMasks, tgtShapeMasks;
    int   resolution=512;
    std::string soundOpt="mute";
    int   audioQuality=30;
    bool  audioHz=false;
    cv::Mat baseImageArray;
    int   fps=30;
    bool* running=nullptr;
    std::function<void(const cv::Mat&,int)> onFrameData;
    std::function<void(int)> onTotal;
};

static std::string imderTimestamp(){
    time_t now2=time(nullptr);
    char ts[32];
    std::tm tms=localTm(now2);
    strftime(ts,sizeof(ts),"%Y%m%d_%H%M%S",&tms);
    return std::string(ts);
}

static void makeProcImages(const ProcessConfig& cfg,cv::Mat& baseImg,cv::Mat& tgtImg,int& W){
    if(cfg.algo=="drawer") baseImg=cfg.baseImageArray.clone();
    else baseImg=readImageSafe(cfg.basePath);
    tgtImg=readImageSafe(cfg.tgtPath);
    if(baseImg.empty()||tgtImg.empty()) throw std::runtime_error("Could not load images");
    baseImg=applyUpscale(baseImg,cfg.resolution);
    tgtImg =applyUpscale(tgtImg, cfg.resolution);
    int limitRes=std::min({baseImg.rows,baseImg.cols,tgtImg.rows,tgtImg.cols});
    int procRes=std::min(cfg.resolution,limitRes);
    if(cfg.algo!="drawer") baseImg=applyTransforms(baseImg,cfg.rotBase,cfg.flipBase);
    tgtImg=applyTransforms(tgtImg,cfg.rotTgt,cfg.flipTgt);
    cv::resize(baseImg,baseImg,cv::Size(procRes,procRes));
    cv::resize(tgtImg, tgtImg, cv::Size(procRes,procRes));
    cv::cvtColor(baseImg,baseImg,cv::COLOR_BGR2RGB);
    cv::cvtColor(tgtImg, tgtImg, cv::COLOR_BGR2RGB);
    W=procRes;
}

static void openExporters(const ProcessConfig& cfg,int W,int H,const std::string& ts,
                          const std::string& outDir,std::string& outPath,std::string& silPath,
                          FfmpegWriter& vw,GIF::Encoder& gifEnc){
    if(cfg.mode=="export_video"){
        outPath=outDir+"/imder_"+ts+".mp4";
        if(cfg.soundOpt!="mute"){silPath=outDir+"/imder_"+ts+"_silent.mp4";vw.open(silPath,W,H,(double)cfg.fps);}
        else vw.open(outPath,W,H,(double)cfg.fps);
        if(!vw.isOpened()) throw std::runtime_error("cannot open video writer - ffmpeg is required for mp4 export");
    } else if(cfg.mode=="export_gif"){
        outPath=outDir+"/imder_"+ts+".gif";
        gifEnc.open(outPath,W,H,1000/cfg.fps);
    }
}

static void finishExport(const ProcessConfig& cfg,const std::string& outPath,const std::string& silPath,
                         const std::vector<cv::Mat>& vframes,const cv::Mat& finalFrame,int W,
                         const std::string& ts,const std::string& outDir,
                         std::function<void(int,const std::string&)>* onProgress,
                         std::function<void(const std::string&)> onFinish,
                         std::function<void(const std::string&)> onError){
    if(cfg.mode=="export_video"){
        if(onProgress) (*onProgress)(99,"muxing audio (ffmpeg)");
        if(!silPath.empty()){
            std::string tgtA=(cfg.soundOpt=="target-sound")?cfg.tgtPath:"";
            std::string fin=addAudioToVideo(silPath,vframes,(double)cfg.fps,outPath,cfg.soundOpt,tgtA,cfg.audioQuality,cfg.audioHz);
            remove(silPath.c_str());
            if(fin==silPath){ onError("audio mux failed - mp4 export produced no output"); return; }
            onFinish("Saved to "+fin);
        } else onFinish("Saved to "+outPath);
    } else if(cfg.mode=="export_image"){
        std::string ip=outDir+"/imder_"+ts+".png";
        if(finalFrame.empty()) throw std::runtime_error("no frame to save");
        cv::Mat bgr;
        cv::cvtColor(finalFrame,bgr,cv::COLOR_RGB2BGR);
        if(!writeImageSafe(ip,bgr)) throw std::runtime_error("failed to write "+ip);
        onFinish("Saved to "+ip);
    } else if(cfg.mode=="export_gif"){
        if(onProgress) (*onProgress)(99,"building gif palette (ffmpeg)");
        onFinish("Saved to "+outPath);
    } else onFinish("Preview finished");
}

static void processCore(const ProcessConfig& cfg,
                        std::function<void(int,const std::string&)> onProgress,
                        std::function<void(const QImage&)> onFrame,
                        std::function<void(const std::string&)> onFinish,
                        std::function<void(const std::string&)> onError)
{
    try {
        static const std::vector<std::string> maskModes={"pattern","disguise","navigate","swap","blend"};
        bool needsMask=std::find(maskModes.begin(),maskModes.end(),cfg.algo)!=maskModes.end();
        if(needsMask&&cfg.mask.empty())
            throw std::runtime_error(cfg.algo+" mode requires analyzing shapes first.");
        if(cfg.algo=="reborn"&&(cfg.baseShapeMasks.empty()||cfg.tgtShapeMasks.empty()))
            throw std::runtime_error("Reborn mode requires analyzed shapes on both base and target.");
        if(cfg.algo=="drawer"&&cfg.baseImageArray.empty())
            throw std::runtime_error("Drawer mode requires base drawing data.");

        const std::string outDir=cfg.outDir.empty()?"results":cfg.outDir;
        QDir().mkpath(QString::fromStdString(outDir));

        if(onProgress) onProgress(0,"loading media");
        cv::Mat baseImg,tgtImg;
        int W=0;
        makeProcImages(cfg,baseImg,tgtImg,W);
        int H=baseImg.rows,N=W*H;
        if(cfg.onTotal) cfg.onTotal(302);
        int totalFrames=302;
        std::string ts=imderTimestamp();

        if(cfg.algo=="reborn"){
            size_t pairs=std::min(cfg.baseShapeMasks.size(),cfg.tgtShapeMasks.size());
            onProgress(2,"matching shapes (reborn)");
            struct RPair{ std::vector<int> sy,sx,ey,ex; std::vector<std::array<float,3>> sc,tc; };
            std::vector<RPair> matched;
            for(size_t k=0;k<pairs;k++){
                cv::Mat bm,tm;
                cv::resize(cfg.baseShapeMasks[k],bm,cv::Size(W,H),0,0,cv::INTER_NEAREST);
                cv::resize(cfg.tgtShapeMasks[k],tm,cv::Size(W,H),0,0,cv::INTER_NEAREST);
                std::vector<int> by,bx,ty,tx;
                for(int y=0;y<H;y++)for(int x=0;x<W;x++){
                    if(bm.at<uint8_t>(y,x)>0){by.push_back(y);bx.push_back(x);}
                    if(tm.at<uint8_t>(y,x)>0){ty.push_back(y);tx.push_back(x);}
                }
                if(by.empty()||ty.empty()) continue;
                size_t bn=by.size(),tn2=ty.size();
                std::vector<uint32_t> bk(bn),tk(tn2);
                for(size_t i=0;i<bn;i++){const uint8_t* p=baseImg.data+(by[i]*W+bx[i])*3;bk[i]=morton3(p[0],p[1],p[2]);}
                for(size_t i=0;i<tn2;i++){const uint8_t* p=tgtImg.data+(ty[i]*W+tx[i])*3;tk[i]=morton3(p[0],p[1],p[2]);}
                std::vector<int> bidx(bn);std::iota(bidx.begin(),bidx.end(),0);
                std::sort(bidx.begin(),bidx.end(),[&](int a,int b){return bk[a]<bk[b];});
                std::vector<int> tidx(tn2);std::iota(tidx.begin(),tidx.end(),0);
                std::sort(tidx.begin(),tidx.end(),[&](int a,int b){return tk[a]<tk[b];});
                int m=(int)std::min(bn,tn2);
                double bcx=0,bcy=0,tcx2=0,tcy2=0;
                for(int i=0;i<m;i++){bcx+=bx[bidx[i]];bcy+=by[bidx[i]];tcx2+=tx[tidx[i]];tcy2+=ty[tidx[i]];}
                bcx/=m;bcy/=m;tcx2/=m;tcy2/=m;
                RPair rp;
                for(int i=0;i<m;i++){
                    const uint8_t* sp=baseImg.data+(by[bidx[i]]*W+bx[bidx[i]])*3;
                    const uint8_t* tp2=tgtImg.data+(ty[tidx[i]]*W+tx[tidx[i]])*3;
                    int ex=std::clamp((int)std::lround(bcx+((double)tx[tidx[i]]-tcx2)),0,W-1);
                    int ey=std::clamp((int)std::lround(bcy+((double)ty[tidx[i]]-tcy2)),0,H-1);
                    rp.sy.push_back(by[bidx[i]]);rp.sx.push_back(bx[bidx[i]]);
                    rp.ey.push_back(ey);rp.ex.push_back(ex);
                    rp.sc.push_back({sp[0]*1.f,sp[1]*1.f,sp[2]*1.f});
                    rp.tc.push_back({tp2[0]*1.f,tp2[1]*1.f,tp2[2]*1.f});
                }
                matched.push_back(std::move(rp));
            }
            if(matched.empty()) throw std::runtime_error("Reborn found no drawable shape pairs.");

            std::string outPath,silPath;
            FfmpegWriter vw; GIF::Encoder gifEnc;
            std::vector<cv::Mat> vframes;
            openExporters(cfg,W,H,ts,outDir,outPath,silPath,vw,gifEnc);
            auto renderReborn=[&](float progress){
                cv::Mat frm=baseImg.clone();
                float t=progress*progress*(3.f-2.f*progress);
                for(auto& rp:matched)
                    for(size_t i=0;i<rp.sy.size();i++){
                        int cy=(int)(rp.sy[i]+(rp.ey[i]-rp.sy[i])*t);
                        int cx=(int)(rp.sx[i]+(rp.ex[i]-rp.sx[i])*t);
                        cy=std::clamp(cy,0,H-1);cx=std::clamp(cx,0,W-1);
                        uint8_t* p=frm.data+(cy*W+cx)*3;
                        p[0]=(uint8_t)std::clamp(rp.sc[i][0]+(rp.tc[i][0]-rp.sc[i][0])*t,0.f,255.f);
                        p[1]=(uint8_t)std::clamp(rp.sc[i][1]+(rp.tc[i][1]-rp.sc[i][1])*t,0.f,255.f);
                        p[2]=(uint8_t)std::clamp(rp.sc[i][2]+(rp.tc[i][2]-rp.sc[i][2])*t,0.f,255.f);
                    }
                return frm;
            };
            for(int f=0;f<totalFrames;f++){
                if(cfg.running&&!*cfg.running) break;
                float progress=(float)f/std::max(1,totalFrames-1);
                onProgress((int)(progress*100),"rendering frame "+std::to_string(f+1)+"/"+std::to_string(totalFrames));
                cv::Mat frm=renderReborn(progress);
                if(cfg.mode=="preview"){
                    QImage qi(frm.data,W,H,W*3,QImage::Format_RGB888);onFrame(qi.copy());
                    if(cfg.onFrameData) cfg.onFrameData(frm,f);
                    std::this_thread::sleep_for(std::chrono::microseconds(16666));
                } else if(cfg.mode=="export_video"){
                    if(cfg.soundOpt!="mute") vframes.push_back(frm.clone());
                    cv::Mat bgr;cv::cvtColor(frm,bgr,cv::COLOR_RGB2BGR);vw.write(bgr);
                    if(cfg.onFrameData) cfg.onFrameData(frm,f);
                } else if(cfg.mode=="export_gif"){
                    gifEnc.writeFrame(frm);
                    if(cfg.onFrameData) cfg.onFrameData(frm,f);
                }
            }
            vw.release();
            finishExport(cfg,outPath,silPath,vframes,renderReborn(1.f),0,ts,outDir,&onProgress,onFinish,onError);
            return;
        }

        if(cfg.algo=="missform"){
            Missform miss(baseImg,tgtImg,127.f);
            std::string outPath,silPath;
            FfmpegWriter vw; GIF::Encoder gifEnc;
            std::vector<cv::Mat> vframes;
            openExporters(cfg,W,H,ts,outDir,outPath,silPath,vw,gifEnc);
            for(int f=0;f<totalFrames;f++){
                if(cfg.running&&!*cfg.running) break;
                float progress=(float)f/(totalFrames-1);
                onProgress((int)(progress*100),"rendering frame "+std::to_string(f+1)+"/"+std::to_string(totalFrames));
                float t=progress*progress*(3.f-2.f*progress);
                cv::Mat frm=miss.frame(t);
                if(cfg.mode=="preview"){
                    QImage qi(frm.data,W,H,W*3,QImage::Format_RGB888);onFrame(qi.copy());
                    if(cfg.onFrameData) cfg.onFrameData(frm,f);
                    std::this_thread::sleep_for(std::chrono::microseconds(16666));
                } else if(cfg.mode=="export_video"){
                    if(cfg.soundOpt!="mute") vframes.push_back(frm.clone());
                    cv::Mat bgr;cv::cvtColor(frm,bgr,cv::COLOR_RGB2BGR);vw.write(bgr);
                    if(cfg.onFrameData) cfg.onFrameData(frm,f);
                } else if(cfg.mode=="export_gif"){
                    gifEnc.writeFrame(frm);
                    if(cfg.onFrameData) cfg.onFrameData(frm,f);
                }
            }
            vw.release();
            finishExport(cfg,outPath,silPath,vframes,miss.frame(1.f),0,ts,outDir,&onProgress,onFinish,onError);
            return;
        }

        cv::Mat procMask;
        if(!cfg.mask.empty()){
            cv::Mat m8; cfg.mask.convertTo(m8,CV_8U,255.0);
            cv::resize(m8,m8,cv::Size(W,H),0,0,cv::INTER_NEAREST);
            procMask=m8.reshape(1,N);
        }

        onProgress(4,"sorting pixels ("+cfg.algo+")");
        std::vector<int32_t> asgn;
        if(cfg.algo=="drawer") asgn=assignDrawerPixels(baseImg,tgtImg);
        else asgn=assignPixels(baseImg,tgtImg,cfg.algo,procMask);

        std::vector<int> validIdx;
        bool allValid=(cfg.algo=="disguise"||cfg.algo=="shuffle"||cfg.algo=="merge"||cfg.algo=="drawer");
        if(allValid){validIdx.resize(N);std::iota(validIdx.begin(),validIdx.end(),0);}
        else for(int i=0;i<N;i++) if(asgn[i]>=0) validIdx.push_back(i);
        int VN=(int)validIdx.size();

        std::vector<int> sX(VN),sY(VN),eX(VN),eY(VN);
        std::vector<std::array<float,3>> srcC(VN),tgtCA(VN);
        const uint8_t* bp2=baseImg.data,*tp2=tgtImg.data;
        for(int i=0;i<VN;i++){
            int vi=validIdx[i];
            sY[i]=vi/W;sX[i]=vi%W;
            int d=asgn[vi];
            eY[i]=pyDiv(d,W);eX[i]=pyMod(d,W);
            srcC[i]={bp2[vi*3+0]*1.f,bp2[vi*3+1]*1.f,bp2[vi*3+2]*1.f};
            if(d>=0) tgtCA[i]={tp2[d*3+0]*1.f,tp2[d*3+1]*1.f,tp2[d*3+2]*1.f};
        }

        std::vector<std::array<float,3>> shuffSrc;
        if(cfg.algo=="fusion"){
            shuffSrc=srcC;
            std::mt19937 rng42(42);
            std::shuffle(shuffSrc.begin(),shuffSrc.end(),rng42);
        }

        std::vector<float> tgtGrayF,srcLuma,posX,posY;
        cv::Mat gradX,gradY;
        if(cfg.algo=="blend"){
            cv::Mat tg; cv::cvtColor(tgtImg,tg,cv::COLOR_RGB2GRAY);
            tg.convertTo(tg,CV_32F,1./255.);
            cv::Sobel(tg,gradX,CV_32F,1,0,3);
            cv::Sobel(tg,gradY,CV_32F,0,1,3);
            tgtGrayF.resize(N);memcpy(tgtGrayF.data(),tg.data,N*sizeof(float));
            srcLuma.resize(VN);
            for(int i=0;i<VN;i++) srcLuma[i]=(0.299f*srcC[i][0]+0.587f*srcC[i][1]+0.114f*srcC[i][2])/255.f;
            posX.resize(VN);posY.resize(VN);
            for(int i=0;i<VN;i++){posX[i]=(float)sX[i];posY[i]=(float)sY[i];}
        }

        std::vector<bool> unmasked(N,false);
        if(cfg.algo=="fusion"&&!procMask.empty())
            for(int i=0;i<N;i++) unmasked[i]=(procMask.data[i]==0);

        std::string outPath,silPath;
        FfmpegWriter vw; GIF::Encoder gifEnc;
        std::vector<cv::Mat> vframes;
        openExporters(cfg,W,H,ts,outDir,outPath,silPath,vw,gifEnc);

        cv::Mat lastFrame;
        for(int f=0;f<totalFrames;f++){
            if(cfg.running&&!*cfg.running) break;
            float progress=(float)f/std::max(1,totalFrames-1);
            onProgress((int)(progress*100),"rendering frame "+std::to_string(f+1)+"/"+std::to_string(totalFrames));
            float t=progress*progress*(3.f-2.f*progress);

            std::vector<int> cx(VN),cy(VN);
            if(cfg.algo=="blend"){
                float kD=0.05f+progress*0.5f,kH=0.05f*(1.f-progress),kG=6.f*(1.f-progress*0.2f);
                for(int i=0;i<VN;i++){
                    int cxi=std::clamp((int)posX[i],0,W-1);
                    int cyi=std::clamp((int)posY[i],0,H-1);
                    float tl=tgtGrayF[cyi*W+cxi];
                    float diff=srcLuma[i]-tl;
                    float gx=((float*)gradX.data)[cyi*W+cxi];
                    float gy=((float*)gradY.data)[cyi*W+cxi];
                    posX[i]+=gx*diff*kG+(sX[i]-posX[i])*kH+(eX[i]-posX[i])*kD;
                    posY[i]+=gy*diff*kG+(sY[i]-posY[i])*kH+(eY[i]-posY[i])*kD;
                    cx[i]=std::clamp((int)posX[i],0,W-1);
                    cy[i]=std::clamp((int)posY[i],0,H-1);
                }
            } else {
                for(int i=0;i<VN;i++){
                    cx[i]=std::clamp((int)(sX[i]+(eX[i]-sX[i])*t),0,W-1);
                    cy[i]=std::clamp((int)(sY[i]+(eY[i]-sY[i])*t),0,H-1);
                }
            }

            cv::Mat frm;
            if(cfg.algo=="disguise"||cfg.algo=="fusion") frm=cv::Mat(H,W,CV_8UC3,cv::Scalar(0,0,0));
            else if(cfg.algo=="drawer")                  frm=cv::Mat(H,W,CV_8UC3,cv::Scalar(255,255,255));
            else if(cfg.algo=="navigate"||cfg.algo=="swap"||cfg.algo=="blend") frm=baseImg.clone();
            else frm=cv::Mat(H,W,CV_8UC3,cv::Scalar(0,0,0));

            if(cfg.algo=="fusion"&&!procMask.empty())
                for(int i=0;i<N;i++) if(unmasked[i]){
                    int y=i/W,x=i%W;
                    uint8_t* p=frm.data+(y*W+x)*3;
                    p[0]=bp2[i*3];p[1]=bp2[i*3+1];p[2]=bp2[i*3+2];
                }

            for(int i=0;i<VN;i++){
                std::array<float,3> c;
                if(cfg.algo=="fusion"){
                    const auto& s2=(cfg.mask.empty()?srcC:shuffSrc)[i];
                    const auto& td=tgtCA[i];
                    c={s2[0]*(1-t)+td[0]*t,s2[1]*(1-t)+td[1]*t,s2[2]*(1-t)+td[2]*t};
                } else c=srcC[i];
                uint8_t* p=frm.data+(cy[i]*W+cx[i])*3;
                p[0]=(uint8_t)std::clamp(c[0],0.f,255.f);
                p[1]=(uint8_t)std::clamp(c[1],0.f,255.f);
                p[2]=(uint8_t)std::clamp(c[2],0.f,255.f);
            }
            lastFrame=frm;

            if(cfg.mode=="preview"){
                QImage qi(frm.data,W,H,W*3,QImage::Format_RGB888);onFrame(qi.copy());
                if(cfg.onFrameData) cfg.onFrameData(frm,f);
                std::this_thread::sleep_for(std::chrono::microseconds(16666));
            } else if(cfg.mode=="export_video"){
                if(cfg.soundOpt!="mute") vframes.push_back(frm.clone());
                cv::Mat bgr;cv::cvtColor(frm,bgr,cv::COLOR_RGB2BGR);vw.write(bgr);
                if(cfg.onFrameData) cfg.onFrameData(frm,f);
            } else if(cfg.mode=="export_gif"){
                gifEnc.writeFrame(frm);
                if(cfg.onFrameData) cfg.onFrameData(frm,f);
            }
        }

        vw.release();
        finishExport(cfg,outPath,silPath,vframes,lastFrame,0,ts,outDir,&onProgress,onFinish,onError);

    } catch(const std::exception& ex){ onError(ex.what()); }
}

static bool isVideoFile(const std::string& p){
    static const char* E[]={"mp4","avi","mov","mkv","flv","wmv"};
    size_t d=p.rfind('.');if(d==std::string::npos)return false;
    std::string e=p.substr(d+1);std::transform(e.begin(),e.end(),e.begin(),::tolower);
    for(auto x:E) if(e==x) return true; return false;
}

static cv::Mat processFramePair(const cv::Mat& baseBGR,const cv::Mat& tgtBGR,
                                const std::string& algo,int resolution){
    int limitRes=std::min({baseBGR.rows,baseBGR.cols,tgtBGR.rows,tgtBGR.cols});
    int procRes=std::min(resolution,limitRes);
    cv::Mat base,tgt;
    cv::resize(baseBGR,base,cv::Size(procRes,procRes));
    cv::resize(tgtBGR,tgt,cv::Size(procRes,procRes));
    cv::cvtColor(base,base,cv::COLOR_BGR2RGB);
    cv::cvtColor(tgt,tgt,cv::COLOR_BGR2RGB);
    if(algo=="missform"){Missform m(base,tgt,127.f);return m.frame(1.f);}
    std::vector<int32_t> asgn;
    if(algo=="drawer") asgn=assignDrawerPixels(base,tgt);
    else asgn=assignPixels(base,tgt,algo,cv::Mat());
    int N=procRes*procRes,W=procRes;
    cv::Mat frm(procRes,procRes,CV_8UC3,cv::Scalar(0,0,0));
    const uint8_t* bp=base.data;
    for(int i=0;i<N;i++){
        if(asgn[i]<0) continue;
        int dy=pyDiv(asgn[i],W),dx=pyMod(asgn[i],W);
        dy=std::clamp(dy,0,procRes-1);dx=std::clamp(dx,0,procRes-1);
        uint8_t* p=frm.data+(dy*W+dx)*3;
        p[0]=bp[i*3];p[1]=bp[i*3+1];p[2]=bp[i*3+2];
    }
    return frm;
}

struct VideoStreamResult {
    std::string mp4Path,gifPath;
};

static void processVideoStream(const std::string& basePath,const std::string& tgtPath,
                               const std::string& algo,int resolution,
                               const std::string& soundOpt,int audioQuality,bool audioHz,
                               const std::string& outDir,bool wantMp4,bool wantGif,bool pace,
                               const std::function<void(int,const std::string&)>& onProgress,
                               const std::function<void(const cv::Mat&,int)>& onFrameData,
                               const std::function<void(int)>& onTotal,
                               const bool* running,VideoStreamResult& result)
{
    if(algo=="fusion") throw std::runtime_error("Fusion algorithm cannot be used with video files.");

    QDir().mkpath(QString::fromStdString(outDir));
    bool bIsVid=isVideoFile(basePath),tIsVid=isVideoFile(tgtPath);

    std::string ts=imderTimestamp();
    std::string videoPath,silPath,gifPath;
    if(wantMp4){
        videoPath=outDir+"/imder_"+ts+".mp4";
        if(soundOpt!="mute") silPath=outDir+"/imder_"+ts+"_silent.mp4";
    }
    if(wantGif) gifPath=outDir+"/imder_"+ts+".gif";

    FfmpegReader rdB,rdT;
    cv::Mat bImg,tImg;
    if(bIsVid){ if(!rdB.open(basePath)) throw std::runtime_error("cannot read base video: "+basePath); }
    else { bImg=readImageSafe(basePath); if(bImg.empty()) throw std::runtime_error("Could not load base image: "+basePath); }
    if(tIsVid){ if(!rdT.open(tgtPath)) throw std::runtime_error("cannot read target video: "+tgtPath); }
    else { tImg=readImageSafe(tgtPath); if(tImg.empty()) throw std::runtime_error("Could not load target image: "+tgtPath); }

    double fps2=bIsVid?rdB.fps:(tIsVid?rdT.fps:30.0);
    if(fps2<=0||fps2>1000) fps2=30.0;

    int bw=bIsVid?rdB.w:bImg.cols, bh=bIsVid?rdB.h:bImg.rows;
    int tw=tIsVid?rdT.w:tImg.cols, th=tIsVid?rdT.h:tImg.rows;
    int limitRes=std::min({bw,bh,tw,th});
    int procRes=std::min(resolution,limitRes);
    if(procRes<1) procRes=1;

    int estTotal=0;
    if(bIsVid&&tIsVid){
        VideoInfo a=probeVideoInfo(basePath),b=probeVideoInfo(tgtPath);
        estTotal=std::min(a.frames,b.frames);
    } else if(bIsVid) estTotal=probeVideoInfo(basePath).frames;
    else if(tIsVid) estTotal=probeVideoInfo(tgtPath).frames;
    if(onTotal) onTotal(std::max(1,estTotal));

    FfmpegWriter vw;
    if(wantMp4){
        if(!vw.open(silPath.empty()?videoPath:silPath,procRes,procRes,fps2))
            throw std::runtime_error("cannot open video writer - ffmpeg is required for mp4 export");
    }
    GIF::Encoder gifEnc;
    if(wantGif) gifEnc.open(gifPath,procRes,procRes,std::max(10,(int)(1000.0/fps2)));

    bool keepFrames=(soundOpt=="sound");
    std::vector<cv::Mat> keepForSound;
    int idx=0;
    cv::Mat bF,tF;
    while(true){
        if(running&&!*running) break;
        if(bIsVid&&!rdB.read(bF)) break;
        if(tIsVid&&!rdT.read(tF)) break;
        cv::Mat& bUse=bIsVid?bF:bImg;
        cv::Mat& tUse=tIsVid?tF:tImg;
        cv::Mat proc=processFramePair(bUse,tUse,algo,resolution);
        if(wantMp4){
            cv::Mat bgr;cv::cvtColor(proc,bgr,cv::COLOR_RGB2BGR);
            vw.write(bgr);
        }
        if(wantGif) gifEnc.writeFrame(proc);
        if(keepFrames) keepForSound.push_back(proc.clone());
        if(onFrameData) onFrameData(proc,idx);
        idx++;
        if(pace) std::this_thread::sleep_for(std::chrono::milliseconds(33));
        if(onProgress){
            std::string stage="rendering frame "+std::to_string(idx);
            int pct=estTotal>0?std::min(99,(int)(100.0*idx/estTotal)):0;
            onProgress(pct,stage);
        }
    }

    rdB.release();rdT.release();
    vw.release();
    if(wantMp4&&soundOpt!="mute"){
        if(onProgress) onProgress(99,"muxing audio (ffmpeg)");
        std::string tgtA=(soundOpt=="target-sound")?tgtPath:"";
        std::string fin=addAudioToVideo(silPath,keepForSound,fps2,videoPath,soundOpt,tgtA,audioQuality,audioHz);
        remove(silPath.c_str());
        if(fin==silPath) throw std::runtime_error("audio mux failed - mp4 export produced no output");
        videoPath=fin;
    }
    if(wantGif){
        if(onProgress) onProgress(99,"building gif palette (ffmpeg)");
        gifEnc.close();
    }
    result.mp4Path=videoPath;
    result.gifPath=gifPath;
}

static bool validateMediaFile(const std::string& path){
    if(!QFile::exists(QString::fromStdString(path))){fprintf(stderr,"Error: File not found: %s\n",path.c_str());return false;}
    static const char* E[]={".png",".jpg",".jpeg",".webp",".mp4",".avi",".mov",".mkv",".flv",".wmv"};
    size_t d=path.rfind('.');if(d==std::string::npos){fprintf(stderr,"Error: No file extension.\n");return false;}
    std::string e=path.substr(d);std::transform(e.begin(),e.end(),e.begin(),::tolower);
    for(auto x:E) if(e==x) return true;
    fprintf(stderr,"Error: Invalid file format. Supported: png jpg jpeg webp mp4 avi mov mkv flv wmv\n");
    return false;
}

static void printProgressBar(int iter,int total,const char* prefix="Progress:",int length=40){
    if(total<=0) return;
    if(iter<0) iter=0;
    if(iter>total) iter=total;
    float pct=100.f*(float)iter/(float)total;
    int filled=(int)((float)length*iter/total);
    std::string bar(filled,'#');
    bar+=std::string(length-filled,'-');
    fprintf(stderr,"\r%s [%s] %3.0f%%",prefix,bar.c_str(),pct);
    if(iter>=total) fprintf(stderr,"\n");
}

static void printBanner(){
    static const char* B[]={
    "██████  ███    ███ ██████  ███████ ██████  ",
    "  ██    ████  ████ ██   ██ ██      ██   ██",
    "  ██    ██ ████ ██ ██   ██ █████   ██████  ",
    "  ██    ██  ██  ██ ██   ██ ██      ██   ██ ",
    "██████  ██      ██ ██████  ███████ ██   ██"};
    printf("\n");
    for(const char* l:B) printf("%s\n",l);
    printf("%s\n",std::string(60,'=').c_str());
    fflush(stdout);
}

static bool cliHasFormat(const std::vector<std::string>& v,const char* f){
    return std::find(v.begin(),v.end(),std::string(f))!=v.end();
}

static std::vector<std::string> cliImageProcess(const std::string& basePath,const std::string& tgtPath,
                                                const std::string& outDir,const std::vector<std::string>& formats,
                                                const std::string& algo,int resolution,
                                                const std::string& soundOpt,int audioQuality,bool audioHz){
    std::vector<std::string> outs;
    bool running=true;
    struct Pass{const char* mode;const char* fmt;};
    std::vector<Pass> want;
    if(cliHasFormat(formats,"png")) want.push_back({"export_image","png"});
    if(cliHasFormat(formats,"gif")) want.push_back({"export_gif","gif"});
    if(cliHasFormat(formats,"mp4")) want.push_back({"export_video","mp4"});
    int passNo=0;
    for(auto& p:want){
        fprintf(stderr,"[imder] pass %d/%d %s\n",++passNo,(int)want.size(),p.fmt);
        ProcessConfig cfg;
        cfg.basePath=basePath;cfg.tgtPath=tgtPath;
        cfg.mode=p.mode;cfg.algo=algo;cfg.outDir=outDir;
        cfg.resolution=resolution;cfg.soundOpt=soundOpt;cfg.audioQuality=audioQuality;cfg.audioHz=audioHz;
        cfg.fps=30;cfg.running=&running;
        processCore(cfg,
            [](int v,const std::string&){ printProgressBar(v,100); },
            [](const QImage&){},
            [&](const std::string& m){
                if(m.rfind("Saved to ",0)==0) outs.push_back(m.substr(9));
            },
            [&](const std::string& e){ fprintf(stderr,"Error: %s\n",e.c_str()); exit(1); });
    }
    return outs;
}

static std::vector<std::string> cliVideoProcess(const std::string& basePath,const std::string& tgtPath,
                                                const std::string& outDir,const std::vector<std::string>& formats,
                                                const std::string& algo,int resolution,
                                                const std::string& soundOpt,int audioQuality,bool audioHz){
    bool wantMp4=cliHasFormat(formats,"mp4"),wantGif=cliHasFormat(formats,"gif");
    VideoStreamResult r;
    processVideoStream(basePath,tgtPath,algo,resolution,soundOpt,audioQuality,audioHz,
                       outDir,wantMp4,wantGif,false,
                       [](int pct,const std::string&){ printProgressBar(pct,100); },
                       {},nullptr,nullptr,r);
    std::vector<std::string> outs;
    if(!r.mp4Path.empty()) outs.push_back(r.mp4Path);
    if(!r.gifPath.empty()) outs.push_back(r.gifPath);
    return outs;
}

static std::vector<std::string> cliProcessAndExport(const std::string& basePath,const std::string& tgtPath,
                                                    const std::string& outDir,const std::vector<std::string>& formats,
                                                    const std::string& algo,int resolution,
                                                    const std::string& soundOpt,int audioQuality,bool audioHz){
    bool bIsVid=isVideoFile(basePath),tIsVid=isVideoFile(tgtPath);
    if(soundOpt=="target-sound"&&!tIsVid){
        fprintf(stderr,"Error: Target sound requires video target\n");exit(1);
    }
    if(bIsVid||tIsVid){
        if(cliHasFormat(formats,"png")){fprintf(stderr,"Error: PNG not supported for video input\n");exit(1);}
        if(algo!="shuffle"&&algo!="merge"&&algo!="missform"){
            fprintf(stderr,"Error: Video only supports: shuffle, merge, missform\n");exit(1);}
        return cliVideoProcess(basePath,tgtPath,outDir,formats,algo,resolution,soundOpt,audioQuality,audioHz);
    }
    if(algo!="shuffle"&&algo!="merge"&&algo!="missform"&&algo!="fusion"){
        fprintf(stderr,"Error: Valid algorithms: shuffle, merge, missform, fusion\n");exit(1);}
    return cliImageProcess(basePath,tgtPath,outDir,formats,algo,resolution,soundOpt,audioQuality,audioHz);
}

struct CliSoundChoice {
    std::string opt="mute";
    int quality=30;
    bool hz=false;
};

static void interactiveCLI(){
    while(true){
        printBanner();
        printf("\n--- Media Selection ---\n");
        fflush(stdout);
        std::string basePath;
        while(true){
            printf("Base: ");fflush(stdout);
            std::string line;std::getline(std::cin,line);
            while(!line.empty()&&(line.back()=='\r'||line.back()==' ')) line.pop_back();
            if(line.empty()){printf("Not found\n");continue;}
            if(QFile::exists(QString::fromStdString(line))&&validateMediaFile(line)){basePath=line;break;}
            printf("Not found\n");
        }
        std::string tgtPath;
        while(true){
            printf("Target: ");fflush(stdout);
            std::string line;std::getline(std::cin,line);
            while(!line.empty()&&(line.back()=='\r'||line.back()==' ')) line.pop_back();
            if(line.empty()){printf("Not found\n");continue;}
            if(QFile::exists(QString::fromStdString(line))&&validateMediaFile(line)){tgtPath=line;break;}
            printf("Not found\n");
        }
        bool bIsVid=isVideoFile(basePath),tIsVid=isVideoFile(tgtPath);

        printf("\nAlgorithm:\n");
        std::vector<std::string> opts;
        if(bIsVid||tIsVid) opts={"merge","shuffle","missform"};
        else opts={"shuffle","merge","missform","fusion"};
        for(size_t i=0;i<opts.size();i++) printf("%zu. %s\n",i+1,opts[i].c_str());
        printf("Select: ");fflush(stdout);
        std::string c;std::getline(std::cin,c);
        std::string algo=opts[0];
        if(!c.empty()&&c.find_first_not_of("0123456789")==std::string::npos){
            int ci=atoi(c.c_str());
            if(ci>=1&&ci<=(int)opts.size()) algo=opts[ci-1];
        }

        printf("\nResolution (1-16384):\n");
        printf("Res: ");fflush(stdout);
        std::string rl;std::getline(std::cin,rl);
        int res=512;
        if(!rl.empty()){
            if(rl.find_first_not_of("0123456789")==std::string::npos){
                res=atoi(rl.c_str());
                if(res<1||res>16384){printf("Invalid resolution, using 512\n");res=512;}
            } else printf("Invalid resolution, using 512\n");
        }

        printf("\nSound (mute/gen%s):\n",tIsVid?"/target":"");
        printf("Sound: ");fflush(stdout);
        std::string sl;std::getline(std::cin,sl);
        std::string snd=sl.empty()?"mute":sl;
        CliSoundChoice sndChoice;
        if(snd!="mute"&&snd!="gen"&&snd!="target"){printf("Invalid sound option '%s', using mute\n",snd.c_str());snd="mute";}
        if(snd=="target"&&!tIsVid){printf("Invalid sound option '%s', using mute\n",snd.c_str());snd="mute";}
        if(snd=="target"){
            printf("\nQuality (sq 1-10 OR sq_hz 8000-192000):\n");
            printf("Quality: ");fflush(stdout);
            std::string ql;std::getline(std::cin,ql);
            if(!ql.empty()&&ql.find_first_not_of("0123456789")==std::string::npos){
                int val=atoi(ql.c_str());
                if(val>=1&&val<=10) sndChoice.quality=val*10;
                else if(val>=8000&&val<=192000) sndChoice.hz=true,sndChoice.quality=val;
                else printf("Invalid quality, using default\n");
            } else if(!ql.empty()) printf("Invalid quality, using default\n");
        }

        printf("\nResults (space separated):\n");
        if(bIsVid||tIsVid) printf("Valid: gif mp4\n");
        else printf("Valid: png gif mp4\n");
        printf("Formats: ");fflush(stdout);
        std::string fl;std::getline(std::cin,fl);
        std::vector<std::string> formats;
        {
            std::istringstream iss(fl);
            std::string tok;
            while(iss>>tok){
                std::transform(tok.begin(),tok.end(),tok.begin(),::tolower);
                formats.push_back(tok);
            }
        }
        if(formats.empty()){printf("No formats specified, using mp4\n");formats={"mp4"};}

        printf("Result folder: ");fflush(stdout);
        std::string ol;std::getline(std::cin,ol);
        std::string outDir=ol.empty()?"results":ol;

        printf("\nProcessing...\n");
        try{
            std::string soundOpt=snd=="gen"?"sound":(snd=="target"?"target-sound":"mute");
            auto files=cliProcessAndExport(basePath,tgtPath,outDir,formats,algo,res,soundOpt,
                                           sndChoice.quality,sndChoice.hz);
            printf("Done:\n");
            for(auto& f:files) printf("  %s\n",f.c_str());
        }catch(const std::exception& e){
            printf("Error: %s\n",e.what());
        }

        printf("\n1. Again\n2. Exit\n");
        while(true){
            printf("Choice: ");fflush(stdout);
            std::string n;std::getline(std::cin,n);
            if(n=="2") return;
            if(n=="1") break;
        }
        printf("\n============================================================\n");
    }
}

class ProcessingThread : public QThread {
    Q_OBJECT
public:
    ProcessConfig cfg;
    QString cacheDir;
    bool _running=true;
    explicit ProcessingThread(const ProcessConfig& c,QObject* p=nullptr):QThread(p),cfg(c){cfg.running=&_running;}
    void stop(){_running=false;}
signals:
    void progressSignal(int,QString);
    void totalSignal(int);
    void frameWritten(int,int,int);
    void finishedSignal(QString);
    void errorSignal(QString);
protected:
    void run() override {
        cfg.onTotal=[this](int t){ emit totalSignal(t); };
        cfg.onFrameData=[this](const cv::Mat& frm,int idx){
            if(cacheDir.isEmpty()) return;
            char name[64];
            snprintf(name,sizeof(name),"frame_%06d.rgb",idx);
            std::string p=cacheDir.toStdString()+"/"+name;
            FILE* f=fopen(p.c_str(),"wb");
            if(f){
                fwrite(frm.data,1,(size_t)frm.total()*frm.elemSize(),f);
                fclose(f);
            }
            emit frameWritten(idx,frm.cols,frm.rows);
        };
        bool anyVid=isVideoFile(cfg.basePath)||isVideoFile(cfg.tgtPath);
        if(anyVid){
            try{
                bool wantMp4=(cfg.mode=="export_video");
                bool wantGif=(cfg.mode=="export_gif");
                VideoStreamResult r;
                processVideoStream(cfg.basePath,cfg.tgtPath,cfg.algo,cfg.resolution,
                                   cfg.soundOpt,cfg.audioQuality,cfg.audioHz,
                                   cfg.outDir,wantMp4,wantGif,!wantMp4&&!wantGif,
                                   [this](int pct,const std::string& st){ emit progressSignal(pct,QString::fromStdString(st)); },
                                   cfg.onFrameData,
                                   [this](int t){ emit totalSignal(t); },
                                   cfg.running,r);
                if(wantMp4) emit finishedSignal(QString::fromStdString("Saved to "+r.mp4Path));
                else if(wantGif) emit finishedSignal(QString::fromStdString("Saved to "+r.gifPath));
                else emit finishedSignal("Preview finished");
            }catch(const std::exception& e){
                emit errorSignal(QString::fromStdString(e.what()));
            }
            return;
        }
        processCore(cfg,
            [this](int v,const std::string& st){ emit progressSignal(v,QString::fromStdString(st)); },
            [this](const QImage&){},
            [this](const std::string& m){ emit finishedSignal(QString::fromStdString(m)); },
            [this](const std::string& e){ emit errorSignal(QString::fromStdString(e)); });
    }
};

class DrawingCanvas : public QLabel {
    Q_OBJECT
public:
    bool drawing=false;
    bool penHasLast=false;
    QPoint lastPt;
    QColor penColor{0,0,0};
    int penWidth=5;
    QSize origSize{1024,1024};

    QImage baseQImage;
    QImage canvasLayer;
    std::vector<QImage> history;
    std::vector<QImage> redoStack;
    static constexpr int MAX_HIST=50;

    explicit DrawingCanvas(QWidget* parent=nullptr):QLabel(parent){
        setSizePolicy(QSizePolicy::Ignored,QSizePolicy::Ignored);
        setMouseTracking(true);
        canvasLayer=QImage(origSize,QImage::Format_ARGB32);
        canvasLayer.fill(Qt::transparent);
        setAlignment(Qt::AlignCenter);
        setStyleSheet("background-color:white;border:1px solid #404040;border-radius:4px;");
        setScaledContents(false);
    }

    void setPenColor(const QColor& c){penColor=c;}
    void setPenWidth(int w){penWidth=std::max(1,std::min(50,w));}

    void updateDisplay(){
        QImage combined;
        if(!baseQImage.isNull()) combined=baseQImage.copy();
        else{ combined=QImage(origSize,QImage::Format_ARGB32); combined.fill(Qt::white); }
        QPainter p(&combined); p.drawImage(0,0,canvasLayer); p.end();
        if(!size().isEmpty()){
            QImage scaled=combined.scaled(size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
            QLabel::setPixmap(QPixmap::fromImage(scaled));
        }
    }

    void resizeEvent(QResizeEvent* e) override { updateDisplay(); QLabel::resizeEvent(e); }

    QPoint getScaledPoint(const QPoint& pos){
        if(pixmap()&&!pixmap()->isNull()){
            QSize ps=pixmap()->size(); QSize ls=size();
            int ox=(ls.width()-ps.width())/2, oy=(ls.height()-ps.height())/2;
            int sx=pos.x()-ox, sy=pos.y()-oy;
            if(sx>=0&&sx<ps.width()&&sy>=0&&sy<ps.height()){
                float scx=(float)origSize.width()/ps.width();
                float scy=(float)origSize.height()/ps.height();
                int cx=std::clamp((int)(sx*scx),0,origSize.width()-1);
                int cy=std::clamp((int)(sy*scy),0,origSize.height()-1);
                return {cx,cy};
            }
        }
        return {-1,-1};
    }

    void mousePressEvent(QMouseEvent* e) override {
        if(e->button()==Qt::LeftButton){
            QPoint p=getScaledPoint(e->pos());
            if(p.x()>=0){ drawing=true; lastPt=p; penHasLast=true; }
            else { drawing=false; penHasLast=false; }
        }
        QLabel::mousePressEvent(e);
    }
    void mouseMoveEvent(QMouseEvent* e) override {
        if(drawing&&penHasLast){
            QPoint p=getScaledPoint(e->pos());
            if(p.x()>=0){ drawLine(lastPt,p); lastPt=p; }
            else penHasLast=false;
        }
        QLabel::mouseMoveEvent(e);
    }
    void mouseReleaseEvent(QMouseEvent* e) override {
        if(e->button()==Qt::LeftButton&&drawing){
            drawing=false; penHasLast=false; saveToHistory();
        }
        QLabel::mouseReleaseEvent(e);
    }

    void drawLine(const QPoint& a,const QPoint& b){
        QPainter p(&canvasLayer);
        p.setPen(QPen(penColor,penWidth,Qt::SolidLine,Qt::RoundCap,Qt::RoundJoin));
        p.drawLine(a,b); p.end(); updateDisplay();
    }

    void saveToHistory(){
        if((int)history.size()>=MAX_HIST) history.erase(history.begin());
        history.push_back(canvasLayer.copy()); redoStack.clear();
    }
    bool undo(){ if(history.size()>1){redoStack.push_back(history.back());history.pop_back();canvasLayer=history.back().copy();updateDisplay();return true;} return false; }
    bool redo(){ if(!redoStack.empty()){history.push_back(redoStack.back());redoStack.pop_back();canvasLayer=history.back().copy();updateDisplay();return true;} return false; }

    void clear(){
        canvasLayer=QImage(origSize,QImage::Format_ARGB32); canvasLayer.fill(Qt::transparent);
        history={canvasLayer.copy()}; redoStack.clear(); updateDisplay();
    }

    cv::Mat getImageArray(){
        QImage combined;
        if(!baseQImage.isNull()) combined=baseQImage.copy();
        else{ combined=QImage(origSize,QImage::Format_ARGB32); combined.fill(Qt::white); }
        QPainter p(&combined); p.drawImage(0,0,canvasLayer); p.end();
        QImage rgb=combined.convertToFormat(QImage::Format_RGB888);
        int w=rgb.width(),h=rgb.height();
        cv::Mat bgr(h,w,CV_8UC3);
        for(int y=0;y<h;y++){
            const uint8_t* src=rgb.constScanLine(y);
            uint8_t* dst=bgr.data+y*w*3;
            for(int x=0;x<w;x++){dst[x*3]=src[x*3+2];dst[x*3+1]=src[x*3+1];dst[x*3+2]=src[x*3];}
        }
        return bgr;
    }

    void setBaseImage(const cv::Mat& imgBGR){
        if(imgBGR.empty()){
            baseQImage=QImage(); origSize=QSize(1024,1024);
            canvasLayer=QImage(origSize,QImage::Format_ARGB32); canvasLayer.fill(Qt::transparent);
            history.clear(); redoStack.clear();
        } else {
            int h=imgBGR.rows,w=imgBGR.cols; origSize=QSize(w,h);
            cv::Mat rgb; cv::cvtColor(imgBGR,rgb,cv::COLOR_BGR2RGB);
            QImage qi(rgb.data,w,h,w*3,QImage::Format_RGB888); baseQImage=qi.copy();
            canvasLayer=QImage(origSize,QImage::Format_ARGB32); canvasLayer.fill(Qt::transparent);
            history={canvasLayer.copy()}; redoStack.clear();
        }
        updateDisplay();
    }
};

class ScalableImageLabel : public QLabel {
    Q_OBJECT
public:
    QPixmap _pix;
    bool drawing=false;
    explicit ScalableImageLabel(QWidget* p=nullptr):QLabel(p){
        setSizePolicy(QSizePolicy::Ignored,QSizePolicy::Ignored);
        setAlignment(Qt::AlignCenter);
    }
    void setPixmap(const QPixmap& pm){ _pix=pm; updateDisplay(); }
    void resizeEvent(QResizeEvent* e) override { updateDisplay(); QLabel::resizeEvent(e); }
    void updateDisplay(){
        if(!_pix.isNull()) QLabel::setPixmap(_pix.scaled(size(),Qt::KeepAspectRatio,Qt::SmoothTransformation));
        else QLabel::setPixmap(QPixmap());
    }
    std::pair<int,int> getCoords(const QPoint& pos){
        if(!_pix.isNull()){
            QPixmap sp=_pix.scaled(size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
            int dx=(size().width()-sp.width())/2, dy=(size().height()-sp.height())/2;
            int cx=pos.x()-dx, cy=pos.y()-dy;
            if(cx>=0&&cx<sp.width()&&cy>=0&&cy<sp.height()){
                float sx=(float)_pix.width()/sp.width();
                float sy=(float)_pix.height()/sp.height();
                return{(int)(cx*sx),(int)(cy*sy)};
            }
        }
        return{-1,-1};
    }
    void mousePressEvent(QMouseEvent* e) override {
        if(e->button()==Qt::LeftButton){
            drawing=true; auto[x,y]=getCoords(e->pos());
            if(x>=0){emit clicked(x,y); emit drawn(x,y);}
        }
        QLabel::mousePressEvent(e);
    }
    void mouseMoveEvent(QMouseEvent* e) override {
        if(drawing){auto[x,y]=getCoords(e->pos());if(x>=0)emit drawn(x,y);}
        QLabel::mouseMoveEvent(e);
    }
    void mouseReleaseEvent(QMouseEvent* e) override {
        if(e->button()==Qt::LeftButton){drawing=false;emit shapeCompleted();}
        QLabel::mouseReleaseEvent(e);
    }
signals:
    void clicked(int,int);
    void drawn(int,int);
    void shapeCompleted();
};

static cv::Mat grabCutRefine(const cv::Mat& bgr,const std::vector<std::pair<int,int>>& pts){
    if(pts.size()<3) return cv::Mat();
    int minx=INT_MAX,miny=INT_MAX,maxx=-1,maxy=-1;
    for(auto& p:pts){
        minx=std::min(minx,p.first);miny=std::min(miny,p.second);
        maxx=std::max(maxx,p.first);maxy=std::max(maxy,p.second);
    }
    int growx=std::max(4,(maxx-minx)/7),growy=std::max(4,(maxy-miny)/7);
    minx=std::max(0,minx-growx);miny=std::max(0,miny-growy);
    maxx=std::min(bgr.cols-1,maxx+growx);maxy=std::min(bgr.rows-1,maxy+growy);
    int rw=maxx-minx+1,rh=maxy-miny+1;
    if(rw<4||rh<4) return cv::Mat();
    double scale=std::min(1.0,640.0/std::max(rw,rh));
    int sw=std::max(2,(int)std::lround(rw*scale)),sh=std::max(2,(int)std::lround(rh*scale));
    cv::Rect roi(minx,miny,rw,rh);
    cv::Mat work;
    cv::resize(bgr(roi),work,cv::Size(sw,sh));
    cv::Mat gmask(sh,sw,CV_8U,cv::Scalar(cv::GC_BGD));
    std::vector<cv::Point> poly;
    for(auto& p:pts)
        poly.push_back(cv::Point(std::clamp((int)std::lround((p.first-minx)*scale),0,sw-1),
                                 std::clamp((int)std::lround((p.second-miny)*scale),0,sh-1)));
    cv::fillPoly(gmask,{poly},cv::Scalar(cv::GC_PR_FGD));
    if(cv::countNonZero(gmask==cv::GC_PR_FGD)==0) return cv::Mat();
    cv::Mat bgd,fgd;
    try{ cv::grabCut(work,gmask,cv::Rect(),bgd,fgd,5,cv::GC_INIT_WITH_MASK); }
    catch(const cv::Exception&){ }
    cv::Mat refined=(gmask==cv::GC_FGD)|(gmask==cv::GC_PR_FGD);
    cv::Mat up;
    cv::resize(refined,up,cv::Size(rw,rh),0,0,cv::INTER_NEAREST);
    cv::Mat out=cv::Mat::zeros(bgr.size(),CV_8U);
    up.copyTo(out(roi));
    return out;
}

class MediaPanel : public QFrame {
    Q_OBJECT
public:
    struct PenShape {
        std::vector<std::pair<int,int>> pts;
        bool plus=true;
    };
    QString filePath, originalFilePath;
    bool isTarget=false;
    int rotateSteps=0; bool isFlipped=false;
    bool isAnalyzing=false;
    cv::Mat segments;
    std::set<int> selectedSegments;
    int procW=256,procH=256;
    QString penMode;
    std::vector<PenShape> shapes;
    std::vector<std::pair<int,int>> currentShape;
    cv::Mat manualMask;
    bool isVideo=false;
    VideoInfo vidInfo;
    cv::Mat firstFrame;
    ScalableImageLabel* preview=nullptr;
    DrawingCanvas* drawingCanvas=nullptr;
    QTemporaryDir drawTmp;
    QWidget* previewContainer=nullptr;
    QLabel* infoLbl=nullptr;
    QPushButton* analyzeBtn=nullptr,*penBtn=nullptr;
    QPushButton* addBtn=nullptr,*removeBtn=nullptr;
    QPushButton* rotateBtn=nullptr,*flipBtn=nullptr;
    QPushButton* clearBtn=nullptr,*resetBtn=nullptr;
    QWidget* drawerTools=nullptr;
    QPushButton* undoBtn=nullptr,*redoBtn=nullptr,*colorBtn=nullptr;
    QSlider* penSizeSlider=nullptr; QLabel* penSizeLbl=nullptr;
    bool drawerMode=false;

    explicit MediaPanel(const QString& title,bool isTgt=false,QWidget* p=nullptr)
        :QFrame(p),isTarget(isTgt){ setupUI(title); }

signals:
    void mediaLoaded(QString);
    void mediaCleared();

public slots:
    void loadMedia(){
        QString f=QFileDialog::getOpenFileName(this,"Open Media","",
            "Media (*.png *.jpg *.jpeg *.webp *.mp4 *.avi *.mov *.mkv *.flv *.wmv)");
        if(!f.isEmpty()){ setMediaData(f,0,false); emit mediaLoaded(f); }
    }
    void clearMedia(){
        filePath.clear();originalFilePath.clear();rotateSteps=0;isFlipped=false;
        isAnalyzing=false;segments=cv::Mat();selectedSegments.clear();
        penMode.clear();shapes.clear();currentShape.clear();manualMask=cv::Mat();
        isVideo=false;vidInfo=VideoInfo();firstFrame=cv::Mat();
        preview->setPixmap(QPixmap());infoLbl->setText("No media loaded");
        addBtn->setText("Add");removeBtn->setEnabled(false);
        rotateBtn->setEnabled(false);flipBtn->setEnabled(false);
        emit mediaCleared();
        if(drawingCanvas){ drawingCanvas->setBaseImage(cv::Mat()); drawingCanvas->clear(); }
    }
    void rotateMedia(){
        if(filePath.isEmpty()||isVideo) return;
        rotateSteps=(rotateSteps+1)%4;updatePreview();
        if(drawerMode&&drawingCanvas) _reloadDrawerBase();
    }
    void flipMedia(){
        if(filePath.isEmpty()||isVideo) return;
        isFlipped=!isFlipped;updatePreview();
        if(drawerMode&&drawingCanvas) _reloadDrawerBase();
    }
    void onAnalyzeClicked(){
        if(filePath.isEmpty()||isVideo) return;
        bool hasPen=!shapes.empty()||!currentShape.empty();
        if(!hasPen){ analyzeShapes(); return; }
        QMenu m;
        m.setStyleSheet(menuStyle());
        tuneMenu(&m);
        m.addAction("As-is",[this]{ analyzeShapes(); });
        m.addAction("Smart",[this]{ analyzeSmart(); });
        m.addSeparator();
        m.addAction("Clear Shapes",[this]{ clearPenShapes(); });
        m.exec(QCursor::pos());
    }
    void analyzeShapes(){
        if(filePath.isEmpty()||isVideo) return;
        try{
            if(!penMode.isEmpty()&&(!shapes.empty()||!currentShape.empty())){
                if(!currentShape.empty()){
                    shapes.push_back({currentShape,penMode=="plus"});
                    currentShape.clear();
                }
                manualMask=composeMask();
                isAnalyzing=true;updatePreview();
                int nInc=0,nExc=0;
                for(auto& s:shapes){ if(s.plus) nInc++; else nExc++; }
                if(nExc>0) infoLbl->setText(QString("%1 include / %2 exclude shapes analyzed.").arg(nInc).arg(nExc));
                else if(nInc<=1) infoLbl->setText("Shape analyzed.");
                else infoLbl->setText(QString("%1 shapes analyzed (combined mask, visually separate).").arg(nInc));
                return;
            }
            cv::Mat img=readImageSafe(filePath.toStdString());
            img=applyTransforms(img,rotateSteps,isFlipped);
            procW=256;procH=256;
            cv::Mat small;cv::resize(img,small,cv::Size(procW,procH));
            cv::Mat blurred; cv::blur(small,blurred,cv::Size(5,5));
            cv::Mat rgb;cv::cvtColor(blurred,rgb,cv::COLOR_BGR2RGB);
            cv::Mat pv=rgb.reshape(1,procW*procH);
            pv.convertTo(pv,CV_32F);
            cv::Mat labels,centers;
            cv::TermCriteria crit(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER,10,1.0);
            cv::kmeans(pv,6,labels,crit,10,cv::KMEANS_RANDOM_CENTERS,centers);
            segments=labels.reshape(1,procH);
            selectedSegments.clear();isAnalyzing=true;
            manualMask=cv::Mat();
            updatePreview();infoLbl->setText("Click shape to select (Green), others (Red)");
        }catch(const std::exception& e){ QMessageBox::warning(this,"Analysis Error",e.what()); }
    }
    void analyzeSmart(){
        if(filePath.isEmpty()||isVideo) return;
        try{
            if(!currentShape.empty()){
                shapes.push_back({currentShape,penMode=="plus"});
                currentShape.clear();
            }
            std::vector<PenShape> incs;
            for(auto& s:shapes) if(s.plus) incs.push_back(s);
            if(incs.empty()){ QMessageBox::warning(this,"Smart Analysis","Draw at least one include (+) shape first."); return; }
            cv::Mat img=readImageSafe(filePath.toStdString());
            img=applyTransforms(img,rotateSteps,isFlipped);
            int h=img.rows,w=img.cols;
            cv::Mat smart=cv::Mat::zeros(h,w,CV_8U);
            int refined=0;
            for(auto& s:incs){
                if((int)s.pts.size()>=3){
                    cv::Mat r=grabCutRefine(img,s.pts);
                    if(!r.empty()){ cv::bitwise_or(smart,r,smart); refined++; continue; }
                    std::vector<cv::Point> poly;
                    for(auto& p:s.pts) poly.push_back(cv::Point(p.first,p.second));
                    cv::fillPoly(smart,{poly},cv::Scalar(255));
                    refined++;
                }
            }
            for(auto& s:shapes) if(!s.plus&&(int)s.pts.size()>=3){
                std::vector<cv::Point> poly;
                for(auto& p:s.pts) poly.push_back(cv::Point(p.first,p.second));
                cv::Mat exc=cv::Mat::zeros(h,w,CV_8U);
                cv::fillPoly(exc,{poly},cv::Scalar(255));
                smart.setTo(0,exc);
            }
            if(cv::countNonZero(smart)==0){ QMessageBox::warning(this,"Smart Analysis","Nothing survived refinement."); return; }
            manualMask=smart;
            isAnalyzing=true;updatePreview();
            infoLbl->setText(QString("Smart analysis: %1 shape(s) refined.").arg(refined));
        }catch(const std::exception& e){ QMessageBox::warning(this,"Analysis Error",e.what()); }
    }
    void clearPenShapes(){
        shapes.clear();currentShape.clear();manualMask=cv::Mat();
        isAnalyzing=false;segments=cv::Mat();selectedSegments.clear();
        updatePreview();
        infoLbl->setText("Shapes cleared.");
    }
    void stopAnalysis(){
        isAnalyzing=false;segments=cv::Mat();selectedSegments.clear();
        penMode.clear();shapes.clear();currentShape.clear();manualMask=cv::Mat();
        updatePreview();
    }
    void setPenMode(const QString& m){
        penMode=m;
        isAnalyzing=false;segments=cv::Mat();selectedSegments.clear();
        updatePreview();
        infoLbl->setText(m=="plus"?"Pen: include (+) - draw shapes, then Analyze":"Pen: exclude (-) - draw shapes, then Analyze");
    }
    void undoDrawing(){ if(drawingCanvas) drawingCanvas->undo(); }
    void redoDrawing(){ if(drawingCanvas) drawingCanvas->redo(); }
    void pickColor(){
        if(!drawingCanvas) return;
        QColor c=QColorDialog::getColor(drawingCanvas->penColor,this);
        if(c.isValid()) drawingCanvas->setPenColor(c);
    }
    void updatePenSize(int v){
        if(drawingCanvas) drawingCanvas->setPenWidth(v);
        penSizeLbl->setText(QString("Size: %1").arg(v));
    }
    void clearDrawing(){ if(drawingCanvas) drawingCanvas->clear(); }
    void resetCanvas(){ if(drawingCanvas){drawingCanvas->setBaseImage(cv::Mat());drawingCanvas->clear();} }

    void onPreviewClicked(int x,int y){
        if(!isAnalyzing) return;
        if(!manualMask.empty()) return;
        if(segments.empty()||preview->_pix.isNull()) return;
        int ow=preview->_pix.width(),oh=preview->_pix.height();
        int sx=std::clamp((int)(x*(float)procW/ow),0,procW-1);
        int sy=std::clamp((int)(y*(float)procH/oh),0,procH-1);
        int lbl=segments.at<int>(sy,sx);
        if(selectedSegments.count(lbl)) selectedSegments.erase(lbl);
        else selectedSegments.insert(lbl);
        updatePreview();
    }
    void onPreviewDrawn(int x,int y){
        if(!penMode.isEmpty()&&!isVideo){ currentShape.push_back({x,y}); updatePreview(); }
    }
    void onShapeCompleted(){
        if(!penMode.isEmpty()&&!isVideo&&!currentShape.empty()){
            shapes.push_back({currentShape,penMode=="plus"});
            currentShape.clear();
            infoLbl->setText(QString("Shape %1 completed. Draw another or click Analyze.").arg((int)shapes.size()));
        }
    }

    void setDrawerMode(bool enabled){
        bool was=drawerMode; drawerMode=enabled;
        if(enabled){
            if(!drawingCanvas){ drawingCanvas=new DrawingCanvas(); }
            auto* lo=qobject_cast<QVBoxLayout*>(previewContainer->layout());
            lo->replaceWidget(preview,drawingCanvas);
            preview->setVisible(false); drawingCanvas->setVisible(true);
            drawingCanvas->clear();
            addBtn->setVisible(false);removeBtn->setVisible(false);
            rotateBtn->setVisible(false);flipBtn->setVisible(false);
            clearBtn->setVisible(true);resetBtn->setVisible(true);
            drawerTools->setVisible(true);infoLbl->setText("Draw on canvas");
            analyzeBtn->setVisible(false);penBtn->setVisible(false);
            if(!filePath.isEmpty()) _reloadDrawerBase();
            else drawingCanvas->setBaseImage(cv::Mat());
        } else {
            if(was&&drawingCanvas){
                cv::Mat drawing=drawingCanvas->getImageArray();
                if(!drawing.empty()){
                    std::string tmp=drawTmp.path().toStdString()+"/drawing.png";
                    writeImageSafe(tmp,drawing);
                    filePath=QString::fromStdString(tmp);
                    originalFilePath=filePath;
                    addBtn->setText("Replace");
                    updatePreview();
                    infoLbl->setText(QString("Drawing (%1x%2)").arg(drawing.cols).arg(drawing.rows));
                }
                auto* lo=qobject_cast<QVBoxLayout*>(previewContainer->layout());
                lo->replaceWidget(drawingCanvas,preview);
                drawingCanvas->setVisible(false);preview->setVisible(true);
                preview->updateDisplay();
            }
            addBtn->setVisible(true);removeBtn->setVisible(true);
            rotateBtn->setVisible(true);flipBtn->setVisible(true);
            clearBtn->setVisible(false);resetBtn->setVisible(false);
            drawerTools->setVisible(false);
            if(filePath.isEmpty()) infoLbl->setText("No media loaded");
        }
    }

    cv::Mat composeMask(){
        if(filePath.isEmpty()||isVideo) return cv::Mat();
        cv::Mat img=readImageSafe(filePath.toStdString());
        if(img.empty()) return cv::Mat();
        int h=img.rows,w=img.cols;
        cv::Mat inc=cv::Mat::zeros(h,w,CV_8U);
        bool any=false;
        for(auto& s:shapes) if(s.plus&&(int)s.pts.size()>=3){
            std::vector<cv::Point> poly;
            for(auto& p:s.pts) poly.push_back(cv::Point(p.first,p.second));
            cv::fillPoly(inc,{poly},cv::Scalar(255));
            any=true;
        }
        if(!segments.empty()&&!selectedSegments.empty()){
            cv::Mat k=cv::Mat::zeros(procH,procW,CV_8U);
            for(int y=0;y<procH;y++)for(int x=0;x<procW;x++)
                if(selectedSegments.count(segments.at<int>(y,x))) k.at<uint8_t>(y,x)=255;
            cv::resize(k,k,cv::Size(w,h),0,0,cv::INTER_NEAREST);
            cv::bitwise_or(inc,k,inc);
            any=true;
        }
        if(!any) return cv::Mat();
        cv::Mat exc=cv::Mat::zeros(h,w,CV_8U);
        for(auto& s:shapes) if(!s.plus&&(int)s.pts.size()>=3){
            std::vector<cv::Point> poly;
            for(auto& p:s.pts) poly.push_back(cv::Point(p.first,p.second));
            cv::fillPoly(exc,{poly},cv::Scalar(255));
        }
        if(cv::countNonZero(exc)>0) inc.setTo(0,exc);
        return inc;
    }

    cv::Mat getMask(){
        if(!manualMask.empty()) return manualMask;
        return composeMask();
    }

    std::vector<cv::Mat> getShapeMasks(){
        std::vector<cv::Mat> out;
        if(filePath.isEmpty()||isVideo) return out;
        cv::Mat img=readImageSafe(filePath.toStdString());
        if(img.empty()) return out;
        int h=img.rows,w=img.cols;
        cv::Mat exc=cv::Mat::zeros(h,w,CV_8U);
        for(auto& s:shapes) if(!s.plus&&(int)s.pts.size()>=3){
            std::vector<cv::Point> poly;
            for(auto& p:s.pts) poly.push_back(cv::Point(p.first,p.second));
            cv::fillPoly(exc,{poly},cv::Scalar(255));
        }
        for(auto& s:shapes) if(s.plus&&(int)s.pts.size()>=3){
            cv::Mat m=cv::Mat::zeros(h,w,CV_8U);
            std::vector<cv::Point> poly;
            for(auto& p:s.pts) poly.push_back(cv::Point(p.first,p.second));
            cv::fillPoly(m,{poly},cv::Scalar(255));
            if(cv::countNonZero(exc)>0) m.setTo(0,exc);
            if(cv::countNonZero(m)>0) out.push_back(m);
        }
        return out;
    }

    bool hasPenShapes() const { return !shapes.empty()||!currentShape.empty(); }

    cv::Mat getDrawingArray(){ return drawingCanvas?drawingCanvas->getImageArray():cv::Mat(); }

    void setMediaData(const QString& path,int rot,bool flip){
        originalFilePath=path; filePath=path;
        rotateSteps=rot; isFlipped=flip;
        isAnalyzing=false;segments=cv::Mat();selectedSegments.clear();
        penMode.clear();shapes.clear();currentShape.clear();manualMask=cv::Mat();
        isVideo=false;vidInfo=VideoInfo();firstFrame=cv::Mat();
        if(!path.isEmpty()){
            if(isVideoFile(path.toStdString())){
                isVideo=true;
                vidInfo=probeVideoInfo(path.toStdString());
                if(vidInfo.ok){
                    firstFrame=decodeWithFfmpeg(path.toStdString());
                    QFileInfo fi(path); double sz=(double)fi.size()/(1024*1024);
                    infoLbl->setText(QString("%1 (%2 MB) %3x%4 %5fps").arg(fi.fileName()).arg(sz,0,'f',1)
                                     .arg(vidInfo.w).arg(vidInfo.h).arg(vidInfo.fps,0,'f',0));
                } else {
                    QFileInfo fi(path); double sz=(double)fi.size()/(1024*1024);
                    infoLbl->setText(QString("%1 (%2 MB) unreadable (ffmpeg needed)").arg(fi.fileName()).arg(sz,0,'f',1));
                }
                rotateBtn->setEnabled(false);flipBtn->setEnabled(false);
            } else {
                cv::Mat img=readImageSafe(path.toStdString());
                QFileInfo fi(path); double sz=(double)fi.size()/(1024*1024);
                if(!img.empty())
                    infoLbl->setText(QString("%1 (%2 MB) %3x%4").arg(fi.fileName()).arg(sz,0,'f',1).arg(img.cols).arg(img.rows));
                else
                    infoLbl->setText(QString("%1 (%2 MB)").arg(fi.fileName()).arg(sz,0,'f',1));
                rotateBtn->setEnabled(true); flipBtn->setEnabled(true);
            }
            addBtn->setText("Replace"); removeBtn->setEnabled(true);
            updatePreview();
            if(drawerMode&&drawingCanvas) _reloadDrawerBase();
        } else clearMedia();
    }

    void updatePreview(){
        if(filePath.isEmpty()) return;
        try{
            cv::Mat img;
            if(isVideo) img=firstFrame.clone();
            else{
                img=readImageSafe(filePath.toStdString());
                if(img.empty()){infoLbl->setText("Error loading image");return;}
                img=applyTransforms(img,rotateSteps,isFlipped);
            }
            if(!isVideo&&isAnalyzing){
                if(!manualMask.empty()){
                    cv::Mat viz=img.clone();
                    cv::Mat overlay=cv::Mat::zeros(img.size(),img.type());
                    cv::Mat maskFull;
                    if(manualMask.rows==img.rows&&manualMask.cols==img.cols) maskFull=manualMask;
                    else cv::resize(manualMask,maskFull,img.size(),0,0,cv::INTER_NEAREST);
                    for(int y=0;y<img.rows;y++) for(int x=0;x<img.cols;x++){
                        bool m=maskFull.at<uint8_t>(y,x)>0;
                        overlay.at<cv::Vec3b>(y,x)=m?cv::Vec3b(0,255,0):cv::Vec3b(255,0,0);
                    }
                    cv::addWeighted(viz,0.7,overlay,0.3,0,viz);img=viz;
                } else if(!segments.empty()){
                    cv::Mat ovl;cv::resize(img,ovl,cv::Size(procW,procH));
                    cv::Mat viz=ovl.clone();
                    auto uq=std::set<int>();
                    for(int y=0;y<procH;y++) for(int x=0;x<procW;x++) uq.insert(segments.at<int>(y,x));
                    for(int lbl:uq){
                        cv::Mat m=cv::Mat::zeros(procH,procW,CV_8U);
                        for(int y=0;y<procH;y++) for(int x=0;x<procW;x++)
                            if(segments.at<int>(y,x)==lbl) m.at<uint8_t>(y,x)=255;
                        std::vector<std::vector<cv::Point>> cnts;
                        cv::findContours(m,cnts,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
                        bool sel=selectedSegments.count(lbl)>0;
                        if(selectedSegments.empty()) cv::drawContours(viz,cnts,-1,cv::Scalar(255,255,0),2);
                        else if(sel) cv::drawContours(viz,cnts,-1,cv::Scalar(0,255,0),2);
                        else cv::drawContours(viz,cnts,-1,cv::Scalar(0,0,255),1);
                    }
                    img=viz;
                }
            }
            if(!isVideo&&(!penMode.isEmpty())&&(!shapes.empty()||!currentShape.empty())){
                cv::Mat viz=img.clone();
                for(auto& sh:shapes){
                    if((int)sh.pts.size()>=2){
                        cv::Scalar col=sh.plus?cv::Scalar(0,255,0):cv::Scalar(0,0,255);
                        std::vector<cv::Point> pts;for(auto[x,y]:sh.pts)pts.push_back({x,y});
                        std::vector<std::vector<cv::Point>> c={pts};
                        cv::polylines(viz,c,false,col,2);
                    }
                }
                if((int)currentShape.size()>=2){
                    cv::Scalar col=(penMode=="plus")?cv::Scalar(0,255,0):cv::Scalar(0,0,255);
                    std::vector<cv::Point> pts;for(auto[x,y]:currentShape)pts.push_back({x,y});
                    std::vector<std::vector<cv::Point>> c={pts};
                    cv::polylines(viz,c,false,col,2);
                }
                img=viz;
            }
            cv::Mat rgb;cv::cvtColor(img,rgb,cv::COLOR_BGR2RGB);
            int w=rgb.cols,h=rgb.rows;
            QImage qi(rgb.data,w,h,w*3,QImage::Format_RGB888);
            preview->setPixmap(QPixmap::fromImage(qi.copy()));
        }catch(const std::exception& e){infoLbl->setText(QString("Error: ")+e.what());}
    }

private:
    void _reloadDrawerBase(){
        cv::Mat img=readImageSafe(filePath.toStdString());
        if(!img.empty()){ img=applyTransforms(img,rotateSteps,isFlipped); drawingCanvas->setBaseImage(img); }
        else drawingCanvas->setBaseImage(cv::Mat());
    }

    void setupUI(const QString& title){
        setStyleSheet(panelStyle()); setObjectName("mediaPanel");
        auto* lo=new QVBoxLayout(this); lo->setContentsMargins(12,12,12,12); lo->setSpacing(10);
        auto* tlbl=new QLabel(title); tlbl->setStyleSheet(titleLblStyle()); tlbl->setAlignment(Qt::AlignCenter);
        lo->addWidget(tlbl);
        auto* toolsLo=new QHBoxLayout();
        analyzeBtn=new QPushButton("Analyze");
        analyzeBtn->setStyleSheet(surfBtnStyle()+"border:1px solid #FFEB3B;color:#FFEB3B;");
        analyzeBtn->setCursor(Qt::PointingHandCursor);
        connect(analyzeBtn,&QPushButton::clicked,this,&MediaPanel::onAnalyzeClicked);
        analyzeBtn->setVisible(false);
        penBtn=new QPushButton("Pen");
        penBtn->setStyleSheet(surfBtnStyle());
        penBtn->setCursor(Qt::PointingHandCursor);
        auto* penMenu=new QMenu();
        penMenu->setStyleSheet(menuStyle());
        tuneMenu(penMenu);
        penMenu->addAction("+ (Include)",[this]{ setPenMode("plus"); });
        penMenu->addAction("- (Exclude)",[this]{ setPenMode("minus"); });
        penMenu->addSeparator();
        penMenu->addAction("Clear Shapes",[this]{ clearPenShapes(); });
        penBtn->setMenu(penMenu); penBtn->setVisible(false);
        toolsLo->addWidget(analyzeBtn); toolsLo->addWidget(penBtn);
        previewContainer=new QWidget();
        auto* pvLo=new QVBoxLayout(previewContainer); pvLo->setContentsMargins(0,0,0,0);
        preview=new ScalableImageLabel();
        preview->setStyleSheet(previewLblStyle()); preview->setMinimumSize(200,200);
        connect(preview,&ScalableImageLabel::clicked,this,&MediaPanel::onPreviewClicked);
        connect(preview,&ScalableImageLabel::drawn,this,&MediaPanel::onPreviewDrawn);
        connect(preview,&ScalableImageLabel::shapeCompleted,this,&MediaPanel::onShapeCompleted);
        pvLo->addWidget(preview);
        lo->addLayout(toolsLo); lo->addWidget(previewContainer,1);
        infoLbl=new QLabel("No media loaded"); infoLbl->setStyleSheet(subtitleLblStyle());
        infoLbl->setAlignment(Qt::AlignCenter); lo->addWidget(infoLbl);
        drawerTools=new QWidget(); auto* dlo=new QGridLayout(drawerTools);
        dlo->setContentsMargins(0,0,0,0); dlo->setSpacing(5);
        undoBtn=new QPushButton("Undo");undoBtn->setStyleSheet(surfBtnStyle());undoBtn->setCursor(Qt::PointingHandCursor);connect(undoBtn,&QPushButton::clicked,this,&MediaPanel::undoDrawing);
        redoBtn=new QPushButton("Redo");redoBtn->setStyleSheet(surfBtnStyle());redoBtn->setCursor(Qt::PointingHandCursor);connect(redoBtn,&QPushButton::clicked,this,&MediaPanel::redoDrawing);
        colorBtn=new QPushButton("Color");colorBtn->setStyleSheet(surfBtnStyle());colorBtn->setCursor(Qt::PointingHandCursor);connect(colorBtn,&QPushButton::clicked,this,&MediaPanel::pickColor);
        penSizeLbl=new QLabel("Size: 5");penSizeLbl->setStyleSheet(subtitleLblStyle());
        penSizeSlider=new QSlider(Qt::Horizontal);penSizeSlider->setRange(1,50);penSizeSlider->setValue(5);
        penSizeSlider->setStyleSheet(sliderStyle());
        connect(penSizeSlider,&QSlider::valueChanged,this,&MediaPanel::updatePenSize);
        dlo->addWidget(undoBtn,0,0);dlo->addWidget(redoBtn,0,1);dlo->addWidget(colorBtn,0,2);
        dlo->addWidget(penSizeLbl,1,0);dlo->addWidget(penSizeSlider,1,1,1,2);
        drawerTools->setVisible(false); lo->addWidget(drawerTools);
        auto* btnLo=new QHBoxLayout();
        addBtn=new QPushButton("Add");addBtn->setStyleSheet(mainBtnStyle());addBtn->setCursor(Qt::PointingHandCursor);connect(addBtn,&QPushButton::clicked,this,&MediaPanel::loadMedia);
        removeBtn=new QPushButton("Remove");removeBtn->setStyleSheet(surfBtnStyle());removeBtn->setCursor(Qt::PointingHandCursor);connect(removeBtn,&QPushButton::clicked,this,&MediaPanel::clearMedia);removeBtn->setEnabled(false);
        btnLo->addWidget(addBtn);btnLo->addWidget(removeBtn);lo->addLayout(btnLo);
        auto* manipLo=new QHBoxLayout();
        rotateBtn=new QPushButton("Rotate");rotateBtn->setStyleSheet(surfBtnStyle());rotateBtn->setCursor(Qt::PointingHandCursor);connect(rotateBtn,&QPushButton::clicked,this,&MediaPanel::rotateMedia);rotateBtn->setEnabled(false);
        flipBtn=new QPushButton("Flip");flipBtn->setStyleSheet(surfBtnStyle());flipBtn->setCursor(Qt::PointingHandCursor);connect(flipBtn,&QPushButton::clicked,this,&MediaPanel::flipMedia);flipBtn->setEnabled(false);
        clearBtn=new QPushButton("Clear");clearBtn->setStyleSheet(surfBtnStyle());clearBtn->setCursor(Qt::PointingHandCursor);connect(clearBtn,&QPushButton::clicked,this,&MediaPanel::clearDrawing);clearBtn->setVisible(false);
        resetBtn=new QPushButton("Reset");resetBtn->setStyleSheet(surfBtnStyle());resetBtn->setCursor(Qt::PointingHandCursor);connect(resetBtn,&QPushButton::clicked,this,&MediaPanel::resetCanvas);resetBtn->setVisible(false);
        manipLo->addWidget(rotateBtn);manipLo->addWidget(flipBtn);manipLo->addWidget(resetBtn);manipLo->addWidget(clearBtn);
        lo->addLayout(manipLo);
    }
};

class ImderGUI : public QMainWindow {
    Q_OBJECT
public:
    QComboBox* modeCombo=nullptr;
    QStandardItemModel* modeModel=nullptr;
    QComboBox* resCombo=nullptr;
    QComboBox* fpsCombo=nullptr;
    QComboBox* soundCombo=nullptr;
    QComboBox* qualityCombo=nullptr;
    MediaPanel* basePanel=nullptr;
    MediaPanel* targetPanel=nullptr;
    QFrame* previewPanel=nullptr;
    ScalableImageLabel* previewDisplay=nullptr;
    QPushButton* reverseBtn=nullptr;
    QPushButton* startBtn=nullptr;
    QPushButton* replayBtn=nullptr;
    QPushButton* stopBtn=nullptr;
    QPushButton* exportBtn=nullptr;
    QLabel* statusBar=nullptr;
    PaintedProgressBar* progress=nullptr;
    QSlider* timeline=nullptr;
    QLabel* frameLbl=nullptr;
    QTimer* pollTimer=nullptr;
    QTimer* playTick=nullptr;
    ProcessingThread* worker=nullptr;
    QString cacheDir;
    int frameW=0,frameH=0;
    int latestIdx=-1,shownIdx=-2,frameCount=0,totalFrames=0;
    int playIdx=0;
    bool playingBack=false,playReverse=false;
    int currentFps=30;

    explicit ImderGUI(QWidget* p=nullptr):QMainWindow(p){
        setWindowTitle("Imder - Image Blender");
        resize(1180,780);
        setStyleSheet(windowStyle());
        QIcon ic(":/imder.png");
        setWindowIcon(ic);
        QApplication::setWindowIcon(ic);

        cacheDir=QStandardPaths::writableLocation(QStandardPaths::CacheLocation)+"/preview";
        clearPreviewCache();

        auto* central=new QWidget(); setCentralWidget(central);
        auto* mainLo=new QVBoxLayout(central); mainLo->setContentsMargins(16,16,16,16); mainLo->setSpacing(12);

        auto* headerLo=new QHBoxLayout(); headerLo->setSpacing(12);
        auto* modeLbl=new QLabel("Mode:"); modeLbl->setStyleSheet(subtitleLblStyle());
        modeCombo=new QComboBox();
        modeModel=new QStandardItemModel(this);
        for(const QString& n:{"Shuffle","Merge","Missform","Fusion","Pattern","Disguise","Navigate","Swap","Blend","Reborn","Drawer"}){
            auto* it=new QStandardItem(n);
            modeModel->appendRow(it);
        }
        modeCombo->setModel(modeModel);
        tuneComboPopup(modeCombo);
        modeCombo->setToolTip("Algorithms gray out based on the loaded media (image or video).");
        modeCombo->setStyleSheet(comboStyle());
        connect(modeCombo,&QComboBox::currentTextChanged,this,&ImderGUI::onModeChanged);
        auto* resLbl=new QLabel("Resolution:"); resLbl->setStyleSheet(subtitleLblStyle());
        resCombo=new QComboBox();
        resCombo->addItems({"128x128","256x256","512x512","768x768","1024x1024","2048x2048","Custom"});
        resCombo->setStyleSheet(comboStyle()); resCombo->setMinimumWidth(120);
        tuneComboPopup(resCombo);
        connect(resCombo,&QComboBox::currentTextChanged,this,&ImderGUI::onResolutionChanged);
        auto* fpsLbl=new QLabel("FPS:"); fpsLbl->setStyleSheet(subtitleLblStyle());
        fpsCombo=new QComboBox();
        fpsCombo->addItems({"30","60","90","120","240"});
        fpsCombo->setCurrentIndex(0); fpsCombo->setStyleSheet(comboStyle()); fpsCombo->setMinimumWidth(80);
        tuneComboPopup(fpsCombo);
        auto* sndLbl=new QLabel("Sound:"); sndLbl->setStyleSheet(subtitleLblStyle());
        soundCombo=new QComboBox();
        soundCombo->addItems({"Mute","Gen","Target"});
        soundCombo->setStyleSheet(comboStyle()); soundCombo->setMinimumWidth(90);
        tuneComboPopup(soundCombo);
        auto* qLbl=new QLabel("Quality:"); qLbl->setStyleSheet(subtitleLblStyle());
        qualityCombo=new QComboBox();
        for(int i=1;i<=10;i++) qualityCombo->addItem(QString::number(i));
        qualityCombo->setCurrentIndex(2);
        qualityCombo->setStyleSheet(comboStyle()); qualityCombo->setMinimumWidth(60);
        tuneComboPopup(qualityCombo);
        qualityCombo->setEnabled(false);
        connect(soundCombo,&QComboBox::currentTextChanged,this,[this](const QString& t){
            qualityCombo->setEnabled(t=="Target");
        });
        headerLo->addWidget(modeLbl);headerLo->addWidget(modeCombo);
        headerLo->addWidget(resLbl);headerLo->addWidget(resCombo);
        headerLo->addWidget(fpsLbl);headerLo->addWidget(fpsCombo);
        headerLo->addWidget(sndLbl);headerLo->addWidget(soundCombo);
        headerLo->addWidget(qLbl);headerLo->addWidget(qualityCombo);
        headerLo->addStretch();
        mainLo->addLayout(headerLo);

        auto* panelsLo=new QHBoxLayout(); panelsLo->setSpacing(16);
        basePanel=new MediaPanel("Base",false);
        previewPanel=new QFrame(); previewPanel->setStyleSheet(panelStyle());
        auto* plo=new QVBoxLayout(previewPanel); plo->setContentsMargins(12,12,12,12); plo->setSpacing(10);
        reverseBtn=new QPushButton("Swap"); reverseBtn->setStyleSheet(surfBtnStyle());
        reverseBtn->setCursor(Qt::PointingHandCursor);
        connect(reverseBtn,&QPushButton::clicked,this,&ImderGUI::swapMedia);
        reverseBtn->setEnabled(false); plo->addWidget(reverseBtn);
        auto* pLbl=new QLabel("Animation Preview"); pLbl->setStyleSheet(titleLblStyle()); pLbl->setAlignment(Qt::AlignCenter);
        plo->addWidget(pLbl);
        previewDisplay=new ScalableImageLabel();
        previewDisplay->setStyleSheet(previewLblStyle()); previewDisplay->setMinimumSize(200,200);
        plo->addWidget(previewDisplay,1);
        auto* tl=new QHBoxLayout(); tl->setSpacing(8);
        frameLbl=new QLabel("frame -/-"); frameLbl->setStyleSheet(subtitleLblStyle());
        frameLbl->setMinimumWidth(110);
        timeline=new QSlider(Qt::Horizontal);
        timeline->setRange(0,0);
        timeline->setStyleSheet(sliderStyle());
        tl->addWidget(frameLbl); tl->addWidget(timeline,1);
        plo->addLayout(tl);
        targetPanel=new MediaPanel("Target",true); targetPanel->setEnabled(false);
        panelsLo->addWidget(basePanel,1); panelsLo->addWidget(previewPanel,1); panelsLo->addWidget(targetPanel,1);
        mainLo->addLayout(panelsLo);

        auto* ctrlLo=new QHBoxLayout(); ctrlLo->setSpacing(12);
        startBtn=new QPushButton("Start Processing"); startBtn->setStyleSheet(mainBtnStyle()); startBtn->setCursor(Qt::PointingHandCursor);
        replayBtn=new QPushButton("Replay"); replayBtn->setStyleSheet(mainBtnStyle()); replayBtn->setCursor(Qt::PointingHandCursor);
        auto* replayMenu=new QMenu(); replayMenu->setStyleSheet(menuStyle());
        tuneMenu(replayMenu);
        replayMenu->addAction("Play Forward",[this]{startPlayback(false);});
        replayMenu->addAction("Play Reverse",[this]{startPlayback(true);});
        replayBtn->setMenu(replayMenu);
        stopBtn=new QPushButton("Stop"); stopBtn->setStyleSheet(surfBtnStyle()); stopBtn->setCursor(Qt::PointingHandCursor);
        exportBtn=new QPushButton("Export"); exportBtn->setStyleSheet(mainBtnStyle()); exportBtn->setCursor(Qt::PointingHandCursor);
        auto* exportMenu=new QMenu(); exportMenu->setStyleSheet(menuStyle());
        tuneMenu(exportMenu);
        exportBtn->setMenu(exportMenu);
        ctrlLo->addWidget(startBtn);ctrlLo->addWidget(replayBtn);ctrlLo->addWidget(stopBtn);ctrlLo->addWidget(exportBtn);
        stopBtn->setEnabled(false); replayBtn->setEnabled(false);
        exportBtn->setEnabled(false); startBtn->setEnabled(false);
        mainLo->addLayout(ctrlLo);

        statusBar=new QLabel("Ready"); statusBar->setStyleSheet("color:#A0A0A0;padding:6px 12px;font-size:12px;");
        mainLo->addWidget(statusBar);
        progress=new PaintedProgressBar();
        progress->setStyleSheet("QProgressBar{border:1px solid #404040;background-color:#1a1a1a;}");
        progress->setValue(0);
        mainLo->addWidget(progress);

        connect(basePanel,&MediaPanel::mediaLoaded,this,&ImderGUI::onBaseLoaded);
        connect(basePanel,&MediaPanel::mediaCleared,this,&ImderGUI::onBaseCleared);
        connect(targetPanel,&MediaPanel::mediaLoaded,this,&ImderGUI::onTargetLoaded);
        connect(targetPanel,&MediaPanel::mediaCleared,this,[this]{checkReady();});
        connect(startBtn,&QPushButton::clicked,this,[this]{startProcess("preview");});
        connect(stopBtn,&QPushButton::clicked,this,&ImderGUI::stopProcess);
        connect(timeline,&QSlider::sliderReleased,this,[this]{
            playIdx=timeline->value();
            showCacheFrame(playIdx);
        });

        pollTimer=new QTimer(this);
        pollTimer->setInterval(33);
        connect(pollTimer,&QTimer::timeout,this,&ImderGUI::pollStreamFrame);
        pollTimer->start();

        playTick=new QTimer(this);
        connect(playTick,&QTimer::timeout,this,&ImderGUI::playbackStep);

        updateModeAvailability();
    }

    ~ImderGUI() override {
        if(worker){worker->stop();worker->wait(3000);}
        clearPreviewCache();
    }

protected:
    void closeEvent(QCloseEvent* e) override {
        if(worker){worker->stop();worker->wait(3000);}
        stopPlayback();
        clearPreviewCache();
        e->accept();
    }

public slots:
    void clearPreviewCache(){
        if(cacheDir.isEmpty()) return;
        QDir d(cacheDir);
        if(d.exists()) d.removeRecursively();
        QDir().mkpath(cacheDir);
        frameW=frameH=0;latestIdx=-1;shownIdx=-2;frameCount=0;playIdx=0;
    }

    void onResolutionChanged(const QString& text){
        if(text=="Custom"){
            bool ok; int v=QInputDialog::getInt(this,"Custom Resolution","Enter resolution (1-16384):",1024,1,16384,1,&ok);
            if(ok){
                QString nr=QString("%1x%1").arg(v);
                int ci=resCombo->findText("Custom");
                int ei=resCombo->findText(nr);
                if(ei>=0) resCombo->setCurrentIndex(ei);
                else{resCombo->insertItem(ci,nr);resCombo->setCurrentIndex(ci);}
            } else resCombo->setCurrentIndex(0);
        }
    }

    void updateModeAvailability(){
        bool hasB=!basePanel->filePath.isEmpty(),hasT=!targetPanel->filePath.isEmpty();
        bool anyVid=basePanel->isVideo||targetPanel->isVideo;
        static const QStringList imageOnly={"Fusion","Pattern","Disguise","Navigate","Swap","Blend","Reborn"};
        for(int r=0;r<modeModel->rowCount();r++){
            auto* it=modeModel->item(r);
            QString name=it->text();
            bool ok=true;
            if(hasB&&hasT){
                if(anyVid&&imageOnly.contains(name)) ok=false;
                if(name=="Drawer"&&targetPanel->isVideo) ok=false;
            }
            it->setEnabled(ok);
            if(!ok&&modeCombo->currentText()==name){
                modeCombo->blockSignals(true);
                modeCombo->setCurrentIndex(modeCombo->findText("Merge"));
                modeCombo->blockSignals(false);
            }
        }
    }

    void onModeChanged(const QString& text){
        QString mode=text.toLower();
        static const QStringList maskModes={"pattern","disguise","navigate","swap","blend"};
        if(mode=="drawer"){
            basePanel->setDrawerMode(true);
            targetPanel->setEnabled(true);
            reverseBtn->setEnabled(false);
            targetPanel->analyzeBtn->setVisible(false);
            targetPanel->penBtn->setVisible(false);
            basePanel->analyzeBtn->setVisible(false);
            basePanel->penBtn->setVisible(false);
            targetPanel->stopAnalysis();
            basePanel->stopAnalysis();
        } else if(mode=="reborn"){
            basePanel->setDrawerMode(false);
            targetPanel->setEnabled(!basePanel->filePath.isEmpty());
            basePanel->analyzeBtn->setVisible(!basePanel->isVideo&&!basePanel->filePath.isEmpty());
            basePanel->penBtn->setVisible(!basePanel->isVideo&&!basePanel->filePath.isEmpty());
            targetPanel->analyzeBtn->setVisible(!targetPanel->isVideo&&!targetPanel->filePath.isEmpty());
            targetPanel->penBtn->setVisible(!targetPanel->isVideo&&!targetPanel->filePath.isEmpty());
        } else {
            basePanel->setDrawerMode(false);
            targetPanel->setEnabled(!basePanel->filePath.isEmpty());
            basePanel->analyzeBtn->setVisible(false);
            basePanel->penBtn->setVisible(false);
            targetPanel->analyzeBtn->setVisible(maskModes.contains(mode)&&!targetPanel->isVideo&&!targetPanel->filePath.isEmpty());
            targetPanel->penBtn->setVisible(maskModes.contains(mode)&&!targetPanel->isVideo&&!targetPanel->filePath.isEmpty());
            if(!maskModes.contains(mode)) targetPanel->stopAnalysis();
        }
        checkReady();
    }
    void swapMedia(){
        QString mode=modeCombo->currentText().toLower();
        if(mode=="drawer") return;
        QString bPath=basePanel->filePath; int bRot=basePanel->rotateSteps; bool bFlip=basePanel->isFlipped;
        QString tPath=targetPanel->filePath; int tRot=targetPanel->rotateSteps; bool tFlip=targetPanel->isFlipped;
        basePanel->setMediaData(tPath,tRot,tFlip);
        targetPanel->setMediaData(bPath,bRot,bFlip);
        updateModeAvailability();
        checkReady();
    }
    void onBaseLoaded(const QString&){ targetPanel->setEnabled(true); updateModeAvailability(); checkReady(); }
    void onBaseCleared(){ targetPanel->setEnabled(false); targetPanel->clearMedia(); updateModeAvailability(); checkReady(); }
    void onTargetLoaded(const QString&){ updateModeAvailability(); checkReady(); }

    void checkReady(){
        QString mode=modeCombo->currentText().toLower();
        bool ready=(mode=="drawer")?!targetPanel->filePath.isEmpty()
        :(!basePanel->filePath.isEmpty()&&!targetPanel->filePath.isEmpty());
        startBtn->setEnabled(ready); exportBtn->setEnabled(ready);
        if(mode!="drawer") reverseBtn->setEnabled(ready);
        bool anyVid=basePanel->isVideo||targetPanel->isVideo;
        if(ready){
            auto* m=qobject_cast<QMenu*>(exportBtn->menu());
            m->clear();
            QAction* frameAct=m->addAction("Frame",[this]{startProcess("export_image");});
            frameAct->setEnabled(!anyVid);
            m->addAction("Animation",[this]{startProcess("export_video");});
            m->addAction("GIF",[this]{startProcess("export_gif");});
        }
    }

    bool validate(){
        QString mode=modeCombo->currentText().toLower();
        if(mode=="drawer") return !targetPanel->filePath.isEmpty();
        if(basePanel->filePath.isEmpty()||targetPanel->filePath.isEmpty()) return false;
        if(soundCombo->currentText()=="Target"&&!targetPanel->isVideo){
            QMessageBox::warning(this,"Sound Option","Target sound requires a video target.");
            return false;
        }
        static const QStringList needShape={"pattern","disguise","navigate","swap","blend"};
        if(needShape.contains(mode)){
            bool hasMask=!targetPanel->getMask().empty();
            bool hasAuto=targetPanel->isAnalyzing&&!targetPanel->selectedSegments.empty();
            if(!hasMask&&!hasAuto){
                QMessageBox::warning(this,"Selection Required",
                QString("For %1 mode, please select a shape on Target image (via Analyze or Pen tool).").arg(mode));
                return false;
            }
        }
        if(mode=="reborn"){
            if(basePanel->getShapeMasks().empty()||targetPanel->getShapeMasks().empty()){
                QMessageBox::warning(this,"Shapes Required",
                "For Reborn mode, draw and analyze at least one shape on the Base and one on the Target.");
                return false;
            }
        }
        return true;
    }

    void startProcess(const QString& mode){
        if(!validate()) return;
        stopPlayback();
        statusBar->setText("Processing...");
        progress->setValue(0); progress->stage="";
        setProcessingState(true);
        clearPreviewCache();
        currentFps=fpsCombo->currentText().toInt();
        ProcessConfig cfg;
        cfg.basePath=basePanel->filePath.toStdString();
        cfg.tgtPath=targetPanel->filePath.toStdString();
        cfg.mode=mode.toStdString();
        cfg.algo=modeCombo->currentText().toLower().toStdString();
        cfg.outDir="results";
        cfg.rotBase=basePanel->rotateSteps; cfg.flipBase=basePanel->isFlipped;
        cfg.rotTgt=targetPanel->rotateSteps; cfg.flipTgt=targetPanel->isFlipped;
        cfg.fps=currentFps;
        cfg.resolution=resCombo->currentText().split('x')[0].toInt();
        QString snd=soundCombo->currentText();
        cfg.soundOpt=snd=="Gen"?"sound":(snd=="Target"?"target-sound":"mute");
        cfg.audioQuality=(qualityCombo->currentText().toInt())*10;
        if(cfg.algo=="drawer"){
            cfg.baseImageArray=basePanel->getDrawingArray();
            if(cfg.baseImageArray.empty()){statusBar->setText("No drawing found");setProcessingState(false);return;}
        } else if(cfg.algo=="reborn"){
            cfg.baseShapeMasks=basePanel->getShapeMasks();
            cfg.tgtShapeMasks=targetPanel->getShapeMasks();
        } else {
            cfg.mask=targetPanel->getMask();
        }
        worker=new ProcessingThread(cfg,this);
        worker->cacheDir=cacheDir;
        connect(worker,&ProcessingThread::progressSignal,this,[this](int v,const QString& st){
            progress->setValue(v);
            progress->stage=st;
            progress->update();
            if(!st.isEmpty()) statusBar->setText(st);
        });
        connect(worker,&ProcessingThread::totalSignal,this,[this](int t){
            totalFrames=t;
            timeline->blockSignals(true);
            timeline->setRange(0,std::max(0,t-1));
            timeline->blockSignals(false);
            frameLbl->setText(QString("frame 0/%1").arg(std::max(0,t)));
        });
        connect(worker,&ProcessingThread::frameWritten,this,[this](int idx,int w,int h){
            latestIdx=idx;frameCount=idx+1;
            if(w>0&&h>0){frameW=w;frameH=h;}
        });
        connect(worker,&ProcessingThread::finishedSignal,this,&ImderGUI::processFinished);
        connect(worker,&ProcessingThread::errorSignal,this,&ImderGUI::processError);
        connect(worker,&QThread::finished,worker,&QObject::deleteLater);
        worker->start();
    }

    void stopProcess(){
        stopPlayback();
        if(worker){worker->stop();statusBar->setText("Stopping...");}
    }

    void pollStreamFrame(){
        if(!worker) return;
        if(latestIdx==shownIdx) return;
        shownIdx=latestIdx;
        showCacheFrame(shownIdx);
    }

    void showCacheFrame(int idx){
        if(frameW<=0||frameH<=0) return;
        char name[64];
        snprintf(name,sizeof(name),"frame_%06d.rgb",idx);
        QFile f(cacheDir+"/"+QString(name));
        if(!f.open(QIODevice::ReadOnly)) return;
        QByteArray data=f.readAll();
        qint64 need=(qint64)frameW*frameH*3;
        if(data.size()<need) return;
        QImage img((const uchar*)data.constData(),frameW,frameH,frameW*3,QImage::Format_RGB888);
        previewDisplay->setPixmap(QPixmap::fromImage(img.copy()));
        timeline->blockSignals(true);
        timeline->setValue(idx);
        timeline->blockSignals(false);
        int tot=totalFrames>0?totalFrames:frameCount;
        frameLbl->setText(QString("frame %1/%2").arg(idx+1).arg(tot));
    }

    void processFinished(const QString& msg){
        statusBar->setText(msg);
        setProcessingState(false);
        progress->setValue(100);
        replayBtn->setEnabled(frameCount>0);
        if(worker){
            worker=nullptr;
            if(shownIdx>=0){ playIdx=shownIdx; }
        }
    }
    void processError(const QString& err){
        statusBar->setText("Error: "+err);
        QMessageBox::critical(this,"Error",err);
        setProcessingState(false);
        worker=nullptr;
    }
    void setProcessingState(bool proc){
        QString mode=modeCombo->currentText().toLower();
        startBtn->setEnabled(!proc); exportBtn->setEnabled(!proc);
        basePanel->setEnabled(!proc); targetPanel->setEnabled(!proc);
        stopBtn->setEnabled(proc);
        if(mode!="drawer") reverseBtn->setEnabled(!proc);
        modeCombo->setEnabled(!proc); resCombo->setEnabled(!proc);
        fpsCombo->setEnabled(!proc); soundCombo->setEnabled(!proc);
        qualityCombo->setEnabled(!proc&&soundCombo->currentText()=="Target");
        replayBtn->setEnabled(false);
    }

    void startPlayback(bool reverse){
        if(frameCount<=0) return;
        playReverse=reverse;
        playingBack=true;
        playIdx=reverse?frameCount-1:0;
        if(playTick){playTick->stop();delete playTick;}
        playTick=new QTimer(this);
        connect(playTick,&QTimer::timeout,this,&ImderGUI::playbackStep);
        playTick->start(std::max(10,1000/currentFps));
        showCacheFrame(playIdx);
    }
    void stopPlayback(){
        playingBack=false;
        if(playTick) playTick->stop();
    }
    void playbackStep(){
        if(!playingBack||frameCount<=0){stopPlayback();return;}
        if(playReverse){ if(playIdx<=0){stopPlayback();return;} playIdx--; }
        else { if(playIdx>=frameCount-1){stopPlayback();return;} playIdx++; }
        showCacheFrame(playIdx);
    }
};


static std::string cliLower(const std::string& s){
    std::string r=s;
    std::transform(r.begin(),r.end(),r.begin(),::tolower);
    return r;
}

static bool cliIsDigits(const std::string& s){
    return !s.empty()&&s.find_first_not_of("0123456789")==std::string::npos;
}

static void cliUsage(){
    fprintf(stderr,"usage: imder [-h] -h, --help show this help message and exit\n");
    fprintf(stderr,"\n");
    fprintf(stderr,"  base            path to the base image or video\n");
    fprintf(stderr,"  target          path to the target image or video\n");
    fprintf(stderr,"  result          result folder for the outputs\n");
    fprintf(stderr,"\n");
    fprintf(stderr,"options:\n");
    fprintf(stderr,"  --results FMT [FMT ...]  output formats, one or more of: png, gif, mp4\n");
    fprintf(stderr,"  --algo ALGO             shuffle | merge | missform | fusion (default: merge)\n");
    fprintf(stderr,"  --res RES               resolution 1-16384 (default: 512)\n");
    fprintf(stderr,"  --sound SOUND           mute | gen | target (default: mute)\n");
    fprintf(stderr,"  --sq SQ                 sound quality 1-10 (target sound)\n");
    fprintf(stderr,"  --sq_hz SQ_HZ           sound sample rate 8000-192000 (target sound)\n");
    fprintf(stderr,"\n");
    fprintf(stderr,"  imder cli               interactive mode\n");
}

static int cliOneShot(int argc,char* argv[]){
    if(argc<5||std::string(argv[4])=="-h"||std::string(argv[4])=="--help"){
        cliUsage();
        return 1;
    }
    std::string basePath=argv[1];
    std::string tgtPath=argv[2];
    std::string outDir=argv[3];
    std::vector<std::string> formats;
    std::string algo="merge";
    int res=512;
    std::string sound="mute";
    bool hasSq=false,hasHz=false;
    int sq=0,hz=0;

    for(int i=4;i<argc;i++){
        std::string a=argv[i];
        if(a=="--results"){
            bool got=false;
            for(int j=i+1;j<argc;j++){
                std::string t=argv[j];
                if(t.rfind("--",0)==0) break;
                formats.push_back(cliLower(t));
                got=true;
                i=j;
            }
            if(!got){fprintf(stderr,"Error: argument --results: expected at least one argument\n");return 1;}
        } else if(a=="--algo"&&i+1<argc){
            algo=cliLower(argv[++i]);
        } else if(a=="--res"&&i+1<argc){
            std::string v=argv[++i];
            if(!cliIsDigits(v)){fprintf(stderr,"Error: argument --res: invalid int value: '%s'\n",v.c_str());return 1;}
            res=atoi(v.c_str());
        } else if(a=="--sound"&&i+1<argc){
            sound=cliLower(argv[++i]);
        } else if(a=="--sq"&&i+1<argc){
            std::string v=argv[++i];
            if(!cliIsDigits(v)){fprintf(stderr,"Error: argument --sq: invalid int value: '%s'\n",v.c_str());return 1;}
            sq=atoi(v.c_str());hasSq=true;
        } else if(a=="--sq_hz"&&i+1<argc){
            std::string v=argv[++i];
            if(!cliIsDigits(v)){fprintf(stderr,"Error: argument --sq_hz: invalid int value: '%s'\n",v.c_str());return 1;}
            hz=atoi(v.c_str());hasHz=true;
        } else if(a=="-h"||a=="--help"){
            cliUsage();
            return 0;
        } else {
            fprintf(stderr,"Error: unrecognized arguments: %s\n",a.c_str());
            return 1;
        }
    }

    if(!QFile::exists(QString::fromStdString(basePath))){fprintf(stderr,"Error: Base file not found: %s\n",basePath.c_str());return 1;}
    if(!QFile::exists(QString::fromStdString(tgtPath))){fprintf(stderr,"Error: Target file not found: %s\n",tgtPath.c_str());return 1;}
    if(!QFile::exists(QString::fromStdString(outDir))) QDir().mkpath(QString::fromStdString(outDir));
    if(formats.empty()){fprintf(stderr,"Error: Results list cannot be empty\n");return 1;}
    for(auto& f:formats){
        if(f!="png"&&f!="gif"&&f!="mp4"){fprintf(stderr,"Error: Invalid format '%s'. Valid: png, gif, mp4\n",f.c_str());return 1;}
    }
    if(res<1||res>16384){fprintf(stderr,"Error: Resolution must be integer between 1 and 16384\n");return 1;}
    if(sound!="mute"&&sound!="gen"&&sound!="target"){fprintf(stderr,"Error: Invalid sound option '%s'. Valid: mute, gen, target\n",sound.c_str());return 1;}
    if(hasSq&&hasHz){fprintf(stderr,"Error: Cannot use both sq and sq_hz. Choose one.\n");return 1;}
    if(hasSq&&(sq<1||sq>10)){fprintf(stderr,"Error: SQ must be integer between 1 and 10\n");return 1;}
    if(hasHz&&sound!="target"){fprintf(stderr,"Error: sq_hz only valid with sound='target'\n");return 1;}
    if(hasHz&&(hz<8000||hz>192000)){fprintf(stderr,"Error: sq_hz must be integer between 8000 and 192000\n");return 1;}

    bool bIsVid=isVideoFile(basePath),tIsVid=isVideoFile(tgtPath);
    if((bIsVid||tIsVid)&&algo!="shuffle"&&algo!="merge"&&algo!="missform"){
        fprintf(stderr,"Error: Video only supports: shuffle, merge, missform\n");return 1;}
    if(!(bIsVid||tIsVid)&&algo!="shuffle"&&algo!="merge"&&algo!="missform"&&algo!="fusion"){
        fprintf(stderr,"Error: Valid algorithms: shuffle, merge, missform, fusion\n");return 1;}
    if((bIsVid||tIsVid)&&cliHasFormat(formats,"png")){fprintf(stderr,"Error: PNG not supported for video input\n");return 1;}
    if(sound=="target"&&!tIsVid){fprintf(stderr,"Error: Target sound requires video target\n");return 1;}

    int audioQuality=hasSq?sq*10:30;
    bool audioHz=hasHz;
    if(hasHz) audioQuality=hz;
    std::string soundOpt=sound=="gen"?"sound":(sound=="target"?"target-sound":"mute");

    auto files=cliProcessAndExport(basePath,tgtPath,outDir,formats,algo,res,soundOpt,audioQuality,audioHz);
    for(auto& f:files) printf("%s\n",f.c_str());
    return 0;
}

int main(int argc,char* argv[]){
    attach_parent_console(argc,argv);
    if(argc==1){
        QApplication app(argc,argv);
        app.setStyle("Fusion");
        applyDarkPalette();
        QApplication::setWindowIcon(QIcon(":/imder.png"));
        ImderGUI win;
        win.show();
        return app.exec();
    }
    std::string a1=argv[1];
    if(a1=="cli"){
        if(argc!=2){
            fprintf(stderr,"Error: cli takes no extra arguments (use one-shot instead)\n");
            return 1;
        }
        interactiveCLI();
        return 0;
    }
    if(a1=="-h"||a1=="--help"||a1=="--version"){
        printf("IMDER v1.3.0\n");
        cliUsage();
        return 0;
    }
    if(argc<4){
        cliUsage();
        return 1;
    }
    return cliOneShot(argc,argv);
}

#include "imder.moc"
