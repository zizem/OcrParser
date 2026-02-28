package org.example.tesseractOCR;

import net.sourceforge.tess4j.Tesseract;
import net.sourceforge.tess4j.TesseractException;

import java.awt.*;
import java.awt.datatransfer.StringSelection;
import java.io.File;

public class FindTempFile {

    final File img = new File(System.getProperty("java.io.tmpdir") + "/ocr_tmp.png"); //Отримуємо тимчасову директорію
    final File flag = new File(img.getPath() + ".ready"); //Зображення + .ready

    public  void findImage() throws InterruptedException, TesseractException {
        //Шукаємо фотографію з затримкою 50 milis
        int maxWait = 100;
        while (!flag.exists()) {
            if (maxWait-- <= 0) throw new InterruptedException("Timeout");
            Thread.sleep(50);
        }
        flag.delete(); //Після знаходження удаляем файл з флагом .ready

        //ініціалізація тесеракту
        Tesseract tess = new Tesseract();
        tess.setDatapath("C:\\Users\\zizem\\OneDrive\\Desktop\\cource\\tessData");
        tess.setLanguage("ukr+rus+eng");
        tess.setPageSegMode(3);
        tess.setVariable("user_defined_dpi", "200");

        String result = tess.doOCR(img);//Передаємо фотографію у тесеракт
        System.out.println(result);
        img.delete(); //видаляємо тимчасове зображення
        StringSelection selection = new StringSelection(result);
        Toolkit.getDefaultToolkit().getSystemClipboard().setContents(selection, null); //сбереження у буфер обміну
    }

}

